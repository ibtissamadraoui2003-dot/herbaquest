"""
HerbaQuest — application de démonstration (Gradio)
==================================================
Scanne une plante, débloque sa carte, gagne de l'XP.

La confiance du modèle (calibrée par température dans le notebook) décide de la mécanique de jeu :
    confiance >= t_high            -> carte débloquée
    t_low <= confiance < t_high    -> mini-quiz d'observation entre les 3 propositions
    confiance < t_low              -> « rescanne sous un autre angle »

Fichiers attendus à côté de ce script :
    export/class_meta.json   (classes, rareté, température, seuils — produit par le notebook)
    export/model.tflite      (ou export/model.keras en secours)
    plant_cards.json         (fiches pédagogiques)

Lancement :  pip install -r requirements.txt  puis  python app.py
"""
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "export"
RARITY_COLOR = {"Commune": "#7a7a75", "Rare": "#2a78d6", "Épique": "#6250d6", "Légendaire": "#c98500"}
QUIZ_XP, OBSERVATION_XP = 5, 5


# ----------------------------------------------------------------------------------------------
# Données : métadonnées du modèle et fiches pédagogiques
# ----------------------------------------------------------------------------------------------
def load_meta():
    path = EXPORT / "class_meta.json"
    if not path.exists():
        raise SystemExit(
            "export/class_meta.json est introuvable.\n"
            "Exécutez le notebook, téléchargez herbaquest_export.zip et décompressez-le dans app/export/."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_card(label, cards):
    """Associe un nom de classe du dataset à sa fiche : le mot-clé le plus long l'emporte
    (« iron cross begonia » doit trouver la fiche du bégonia croix de fer, pas celle du bégonia)."""
    name, best, best_len = normalize(label), None, 0
    for card in cards:
        for key in card["match"]:
            if key in name and len(key) > best_len:
                best, best_len = card, len(key)
    if best is None:  # classe sans fiche : carte générique, l'application reste utilisable
        best = {"nom": label.split("(")[0].strip(), "scientifique": "—", "famille": "—", "origine": "—",
                "toxique_chat": None, "toxicite": "Information non renseignée.",
                "indice": "Observe la forme et la disposition des feuilles.", "fait": "Fiche à compléter."}
    return best


# ----------------------------------------------------------------------------------------------
# Modèle : même prétraitement que dans le notebook (petit côté -> 256, recadrage central 224)
# ----------------------------------------------------------------------------------------------
def preprocess(image, img_size=224, resize_to=256):
    im = ImageOps.exif_transpose(image).convert("RGB")
    scale = resize_to / min(im.size)
    im = im.resize((max(resize_to, round(im.width * scale)), max(resize_to, round(im.height * scale))), Image.BILINEAR)
    left, top = (im.width - img_size) // 2, (im.height - img_size) // 2
    return np.asarray(im.crop((left, top, left + img_size, top + img_size)), dtype=np.float32)  # pixels 0–255


def load_predictor():
    """Renvoie une fonction image(224,224,3) -> probabilités. TFLite si possible, Keras sinon."""
    tflite_path = EXPORT / "model.tflite"
    if tflite_path.exists():
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
        interpreter = Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        i = interpreter.get_input_details()[0]["index"]
        o = interpreter.get_output_details()[0]["index"]

        def run(x):
            interpreter.set_tensor(i, x[None].astype(np.float32))
            interpreter.invoke()
            return interpreter.get_tensor(o)[0]
        return run
    import tensorflow as tf
    model = tf.keras.models.load_model(str(EXPORT / "model.keras"))
    return lambda x: model.predict(x[None], verbose=0)[0]


def calibrate(probs, temperature):
    """Mise à l'échelle en température appliquée aux probabilités du modèle."""
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ----------------------------------------------------------------------------------------------
# Logique de jeu (indépendante de l'interface : testable seule)
# ----------------------------------------------------------------------------------------------
def new_state():
    return {"xp": 0, "cards": {}, "candidates": [], "quiz": None}


def decide(probs, meta):
    """Traduit les probabilités calibrées en décision de jeu."""
    top = [int(i) for i in np.argsort(-probs)[:3]]
    confidence = float(probs[top[0]])
    zone = "carte" if confidence >= meta["t_high"] else "quiz" if confidence >= meta["t_low"] else "rescan"
    return zone, top, confidence


def unlock(state, label, meta, confirmed=True):
    """Ajoute une carte à la collection. Renvoie l'XP gagné (0 si la carte était déjà possédée)."""
    already = label in state["cards"]
    if already and (state["cards"][label]["confirmed"] or not confirmed):
        return 0
    rarity = meta["rarity"].get(label, "Commune")
    gain = meta["xp"].get(rarity, 10)
    if already:            # carte « à confirmer » qui devient confirmée : on verse le complément
        gain -= gain // 2
    elif not confirmed:
        gain //= 2
    state["cards"][label] = {"confirmed": confirmed}
    state["xp"] += gain
    return gain


REGIONS = {  # grandes régions, pour que les mauvaises réponses ne soient jamais « à moitié vraies »
    "Amériques": ("amérique", "mexique", "brésil", "colombie", "panama", "guatemala", "caroline", "amériques"),
    "Afrique": ("afrique", "madagascar", "tanzanie", "mozambique"),
    "Asie et Océanie": ("asie", "chine", "japon", "taïwan", "vietnam", "polynésie", "australie", "inde"),
    "Europe et Méditerranée": ("europe", "méditerranée", "arabique", "nord tempéré"),
}


def regions_of(origin):
    o = origin.lower()
    return {r for r, keys in REGIONS.items() if any(k in o for k in keys)}


def overlaps(field, a, b):
    """Vrai si la réponse a pourrait aussi être juste pour b (même région, ou famille en commun)."""
    if a == b:
        return True
    if field == "origine":
        ra, rb = regions_of(a), regions_of(b)
        return not ra or not rb or bool(ra & rb)
    fa = {x.strip().lower() for x in a.split("/")}
    fb = {x.strip().lower() for x in b.split("/")}
    return bool(fa & fb)


def make_quiz(label, cards, rng=random):
    """Question de connaissance tirée de la fiche : la bonne réponse est connue, on peut la corriger."""
    card = find_card(label, cards)
    kinds = [k for k in ("toxique", "famille", "origine")
             if not (k == "toxique" and card["toxique_chat"] is None) and card.get("famille", "—") != "—"]
    if not kinds:
        return None
    kind = rng.choice(kinds)
    if kind == "toxique":
        return {"question": f"La plante « {card['nom']} » est-elle toxique pour les chats ?",
                "choices": ["Oui", "Non"], "answer": "Oui" if card["toxique_chat"] else "Non",
                "explain": card["toxicite"]}
    field = "famille" if kind == "famille" else "origine"
    others = sorted({c[field] for c in cards if not overlaps(field, c[field], card[field])})
    if len(others) < 2:
        return None
    choices = rng.sample(others, 2) + [card[field]]
    rng.shuffle(choices)
    text = "À quelle famille botanique appartient" if kind == "famille" else "De quelle région est originaire"
    return {"question": f"{text} « {card['nom']} » ?", "choices": choices, "answer": card[field],
            "explain": f"{card['nom']} ({card['scientifique']}) — {card[field]}."}


# ----------------------------------------------------------------------------------------------
# Rendu HTML
# ----------------------------------------------------------------------------------------------
CSS = """
.hq-hud{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 8px}
.hq-pill{background:#12372a;color:#f4f1e8;border-radius:999px;padding:6px 14px;font-weight:600;font-size:14px}
.hq-card{border:3px solid var(--rar);border-radius:16px;padding:16px 18px;background:#fbfaf5;color:#16241c;
         box-shadow:0 4px 14px rgba(0,0,0,.08)}
.hq-card h3{margin:0 0 2px;font-size:22px;color:#12372a}
.hq-card .sci{font-style:italic;color:#4d5a52;margin-bottom:8px}
.hq-card .tag{display:inline-block;background:var(--rar);color:#fff;border-radius:6px;padding:2px 8px;
              font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.hq-card dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:10px 0 0;font-size:14px}
.hq-card dt{font-weight:700;color:#12372a}.hq-card dd{margin:0}
.hq-tox{color:#b3261e;font-weight:700}.hq-safe{color:#1e6b3a;font-weight:700}
.hq-msg{border-radius:12px;padding:12px 16px;font-size:16px;font-weight:600;background:#eef4ee;color:#12372a}
.hq-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.hq-mini{border:2px solid var(--rar);border-radius:12px;padding:10px;background:#fbfaf5;color:#16241c;font-size:13px}
.hq-mini.locked{border-color:#d5d3cb;color:#a3a199;background:#f1efe8}
.hq-mini b{display:block;font-size:14px}
/* Le thème sombre de Gradio impose une couleur de texte claire : on la remplace dans les cartes (fond clair). */
.hq-card dd,.hq-card p,.hq-card span:not(.tag):not(.hq-tox):not(.hq-safe){color:#2c3a31 !important}
.hq-card .sci{color:#4d5a52 !important}
.hq-card h3,.hq-card dt{color:#12372a !important}
.hq-card .tag{color:#ffffff !important}
.hq-card .hq-tox{color:#b3261e !important}.hq-card .hq-safe{color:#1e6b3a !important}
.hq-mini,.hq-mini b,.hq-mini i{color:#16241c !important}
.hq-mini.locked,.hq-mini.locked b{color:#8a887f !important}
.hq-msg{color:#12372a !important}
"""


def render_card(label, cards, meta, pending=False):
    c = find_card(label, cards)
    rarity = meta["rarity"].get(label, "Commune")
    tox = ("<span class='hq-tox'>⚠ Toxique pour les chats</span>" if c["toxique_chat"]
           else "<span class='hq-safe'>✓ Sans danger pour les chats</span>" if c["toxique_chat"] is False else "—")
    suffix = " · à confirmer" if pending else ""
    return (f"<div class='hq-card' style='--rar:{RARITY_COLOR.get(rarity, '#7a7a75')}'>"
            f"<span class='tag'>{rarity}{suffix}</span><h3>{c['nom']}</h3><div class='sci'>{c['scientifique']}</div>"
            f"<dl><dt>Famille</dt><dd>{c['famille']}</dd><dt>Origine</dt><dd>{c['origine']}</dd>"
            f"<dt>Chats</dt><dd>{tox}<br>{c['toxicite']}</dd>"
            f"<dt>Reconnaître</dt><dd>{c['indice']}</dd><dt>Le saviez-vous ?</dt><dd>{c['fait']}</dd></dl></div>")


def render_hud(state, meta):
    owned, total = len(state["cards"]), len(meta["classes"])
    return (f"<div class='hq-hud'><span class='hq-pill'>⭐ Niveau {1 + state['xp'] // 100}</span>"
            f"<span class='hq-pill'>✨ {state['xp']} XP</span>"
            f"<span class='hq-pill'>🃏 {owned} / {total} cartes</span></div>")


def render_collection(state, cards, meta):
    order = {"Légendaire": 0, "Épique": 1, "Rare": 2, "Commune": 3}
    items = []
    for label in sorted(meta["classes"], key=lambda l: (order.get(meta["rarity"].get(l, "Commune"), 3), l)):
        rarity = meta["rarity"].get(label, "Commune")
        if label in state["cards"]:
            c = find_card(label, cards)
            note = "" if state["cards"][label]["confirmed"] else " (à confirmer)"
            items.append(f"<div class='hq-mini' style='--rar:{RARITY_COLOR.get(rarity)}'><b>{c['nom']}</b>"
                         f"<i>{c['scientifique']}</i><br>{rarity}{note}</div>")
        else:
            items.append(f"<div class='hq-mini locked'><b>???</b>{rarity}<br>À découvrir</div>")
    return f"<div class='hq-grid'>{''.join(items)}</div>"


def message(text):
    return f"<div class='hq-msg'>{text}</div>"


# ----------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------
def build_app():
    import gradio as gr

    meta = load_meta()
    cards = json.loads((HERE / "plant_cards.json").read_text(encoding="utf-8"))["cards"]
    classes = meta["classes"]
    predict = load_predictor()
    missing = [c for c in classes if "match" not in find_card(c, cards)]
    if missing:
        print(f"⚠ {len(missing)} classe(s) sans fiche pédagogique : {missing}")

    def display_name(label):
        return find_card(label, cards)["nom"]

    def quiz_updates(state, label):
        quiz = make_quiz(label, cards)
        state["quiz"] = quiz
        if quiz is None:
            return gr.update(visible=False), gr.update(), gr.update(value="")
        return (gr.update(visible=True), gr.update(choices=quiz["choices"], value=None, label=quiz["question"]),
                gr.update(value=""))

    def scan(image, state):
        hidden = gr.update(visible=False)
        if image is None:
            return (state, message("📷 Prends ou charge une photo de plante."), "", {}, hidden, gr.update(),
                    hidden, gr.update(), gr.update(), render_hud(state, meta), render_collection(state, cards, meta))
        probs = calibrate(predict(preprocess(image, meta["img_size"], meta["resize_to"])), meta["temperature"])
        zone, top, conf = decide(probs, meta)
        scores = {display_name(classes[i]): float(probs[i]) for i in top}
        state["candidates"], state["quiz"] = [classes[i] for i in top], None
        obs, obs_radio, quiz_box, quiz_radio, quiz_fb = hidden, gr.update(), hidden, gr.update(), gr.update(value="")
        if zone == "carte":
            label = classes[top[0]]
            gain = unlock(state, label, meta)
            msg = message(f"🃏 Carte débloquée ! +{gain} XP" if gain else "🃏 Tu possèdes déjà cette carte.")
            card = render_card(label, cards, meta)
            quiz_box, quiz_radio, quiz_fb = quiz_updates(state, label)
        elif zone == "quiz":
            msg = message("❓ J'hésite entre trois espèces. Compare les indices ci-dessous avec ta plante, puis choisis.")
            card = "".join(f"<p><b>{display_name(l)}</b> — {find_card(l, cards)['indice']}</p>" for l in state["candidates"])
            obs = gr.update(visible=True)
            obs_radio = gr.update(choices=[display_name(l) for l in state["candidates"]], value=None)
        else:
            msg = message(f"🔄 Je ne suis pas assez sûr ({conf:.0%}). Rapproche-toi, cadre les feuilles et rescanne.")
            card = ""
        return (state, msg, card, scores, obs, obs_radio, quiz_box, quiz_radio, quiz_fb,
                render_hud(state, meta), render_collection(state, cards, meta))

    def choose(choice, state):
        label = next((l for l in state["candidates"] if display_name(l) == choice), None)
        if label is None:
            return (state, message("Choisis l'une des trois propositions."), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), render_hud(state, meta), render_collection(state, cards, meta))
        gain = unlock(state, label, meta, confirmed=False)
        state["xp"] += OBSERVATION_XP
        msg = message(f"🔎 Bien observé ! Carte ajoutée « à confirmer » : +{gain + OBSERVATION_XP} XP. "
                      "Rescanne sous un meilleur angle pour la confirmer.")
        quiz_box, quiz_radio, quiz_fb = quiz_updates(state, label)
        return (state, msg, render_card(label, cards, meta, pending=True), gr.update(visible=False), quiz_box, quiz_radio,
                quiz_fb, render_hud(state, meta), render_collection(state, cards, meta))

    def answer(choice, state):
        quiz = state.get("quiz")
        if not quiz:                      # question déjà répondue (second clic) : on garde le résultat affiché
            return state, gr.update(), render_hud(state, meta)
        if choice is None:
            return state, "Choisis une réponse.", render_hud(state, meta)
        state["quiz"] = None
        if choice == quiz["answer"]:
            state["xp"] += QUIZ_XP
            return state, f"✅ Exact ! +{QUIZ_XP} XP — {quiz['explain']}", render_hud(state, meta)
        return state, f"❌ Pas tout à fait : la bonne réponse était « {quiz['answer']} ». {quiz['explain']}", render_hud(state, meta)

    with gr.Blocks(title="HerbaQuest", css=CSS) as app:
        state = gr.State(new_state())
        gr.Markdown("# 🌿 HerbaQuest\nScanne une plante, débloque sa carte, complète ton herbier.")
        hud = gr.HTML(render_hud(new_state(), meta))
        with gr.Tab("📷 Scanner"):
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(type="pil", label="Photo de la plante", height=340)
                    scan_btn = gr.Button("Scanner", variant="primary")
                    scores = gr.Label(num_top_classes=3, label="Propositions du modèle (confiance calibrée)")
                with gr.Column(scale=1):
                    status = gr.HTML(message("📷 Prends ou charge une photo de plante."))
                    card = gr.HTML()
                    with gr.Group(visible=False) as obs_box:
                        obs_radio = gr.Radio([], label="Quelle plante as-tu devant toi ?")
                        obs_btn = gr.Button("Valider mon observation")
                    with gr.Group(visible=False) as quiz_box:
                        quiz_radio = gr.Radio([], label="Question bonus")
                        quiz_btn = gr.Button("Répondre")
                        quiz_fb = gr.Markdown()
        with gr.Tab("🃏 Collection"):
            collection = gr.HTML(render_collection(new_state(), cards, meta))
        gr.Markdown("<small>Projet académique — Ibtissam Draoui, Aya Laazizi, Majda Bakhari — Master IADAIG, ENS de Fès. Les informations de toxicité sont "
                    "indicatives et ne remplacent pas l'avis d'un vétérinaire.</small>")

        scan_btn.click(scan, [image, state], [state, status, card, scores, obs_box, obs_radio, quiz_box, quiz_radio,
                                              quiz_fb, hud, collection])
        obs_btn.click(choose, [obs_radio, state], [state, status, card, obs_box, quiz_box, quiz_radio, quiz_fb, hud, collection])
        quiz_btn.click(answer, [quiz_radio, state], [state, quiz_fb, hud])
    return app


if __name__ == "__main__":
    build_app().launch()
