"""Generate synthetic placeholder images for the benchmark sample data."""
import os

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    exit(1)

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "sample_data", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def make_image(name: str, width: int, height: int, bg_color: str, text_lines: list[str], text_color: str = "black"):
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = font
    y = 20
    for i, line in enumerate(text_lines):
        f = font if i == 0 else font_small
        draw.text((20, y), line, fill=text_color, font=f)
        y += 28
    path = os.path.join(IMAGES_DIR, name)
    img.save(path)
    print(f"  Created {path}")

print("Generating placeholder images...")

make_image("timetable_zurich.png", 600, 400, "#f0f0f0", [
    "Zürich HB - Abfahrtstafel",
    "─────────────────────────",
    "08:32  Bern         IC1   Gleis 7",
    "08:35  Luzern       IR    Gleis 3",
    "08:37  Basel SBB    ICE   Gleis 11",
    "08:40  Winterthur   S8    Gleis 44",
    "08:45  Chur         IC2   Gleis 9",
    "",
    "Gleis = Platform",
])

make_image("timetable_geneva.png", 600, 400, "#f5f5f5", [
    "Genève-Cornavin - Tableau",
    "─────────────────────────",
    "09:10  Lausanne    IR    Voie 3",
    "09:15  Bern        IC1   Voie 5",
    "09:20  Zürich HB   IC1   Voie 7",
    "09:25  Marseille   TGV   Voie 1",
])

make_image("timetable_bern.png", 600, 400, "#f0f0f0", [
    "Bern - Abfahrtstafel",
    "─────────────────────────",
    "10:05  Luzern       IR    1h05  Gleis 4",
    "10:10  Zürich HB    IC1   0h56  Gleis 7",
    "10:15  Interlaken   RE    0h47  Gleis 3",
    "10:20  Basel        ICE   0h52  Gleis 9",
])

make_image("timetable_basel.png", 600, 400, "#f5f5f5", [
    "Basel SBB - Departures",
    "─────────────────────────",
    "11:00  Zürich HB    IC1   Gleis 6",
    "11:05  Bern         IC1   Gleis 3",
    "11:10  Interlaken   IR    Gleis 8",
    "11:15  Luzern       IR    Gleis 2",
    "11:20  Geneva       IC1   Gleis 5",
])

make_image("timetable_lugano.png", 600, 400, "#f0f0f0", [
    "Lugano - Partenze",
    "─────────────────────────",
    "12:00  Milano Centrale  EC 17   Bin. 2",
    "12:05  Bellinzona       S10     Bin. 1",
    "12:10  Zürich HB        IC2     Bin. 4",
    "12:15  Locarno          S20     Bin. 3",
])

make_image("receipt_migros.png", 400, 500, "#ffffff", [
    "MIGROS",
    "Bahnhofstrasse 10, Zürich",
    "─────────────────────────",
    "Vollmilch 1L        CHF 1.50",
    "Brot geschnitten    CHF 3.20",
    "Butter 250g         CHF 2.80",
    "Eier 10er           CHF 5.30",
    "Apfel 1kg           CHF 3.50",
    "─────────────────────────",
    "Subtotal             CHF 16.30",
    "Mehrwertsteuer 2.5%  CHF 0.41",
    "TOTAL                CHF 47.30",
    "─────────────────────────",
    "Kartenzahlung",
    "Vielen Dank!",
])

make_image("receipt_coop.png", 400, 500, "#ffffff", [
    "COOP",
    "Rue du Rhône 50, Genève",
    "─────────────────────────",
    "Fromage Gruyère     CHF 12.50",
    "Pain complet         CHF 4.20",
    "Tomates 500g         CHF 3.80",
    "─────────────────────────",
    "Sous-total           CHF 20.50",
    "TVA 7.7%             CHF 1.58",
    "TOTAL                CHF 32.50",
    "─────────────────────────",
    "Paiement par carte",
    "Merci!",
])

make_image("receipt_denner.png", 400, 500, "#ffffff", [
    "DENNER",
    "Langstrasse 5, Zürich",
    "─────────────────────────",
    "Bier 6x0.33L        CHF 8.90",
    "Chips 200g           CHF 2.50",
    "Schokolade 100g      CHF 1.80",
    "Wasser 1.5L          CHF 0.90",
    "─────────────────────────",
    "Subtotal             CHF 14.10",
    "MWST 2.5%            CHF 2.48",
    "TOTAL                CHF 19.95",
])

make_image("swiss_landscape.png", 800, 500, "#87CEEB", [
    "Matterhorn at Sunrise",
    "",
    "     /\\",
    "    /  \\",
    "   /    \\   Snow-capped peaks",
    "  /      \\",
    " /        \\",
    "/__________\\",
    " Green alpine meadows",
])

make_image("zurich_street.png", 800, 500, "#D3D3D3", [
    "Bahnhofstrasse, Zürich",
    "",
    "  [Tram]  [Shops]  [People]",
    "",
    "  Luxury boutiques along",
    "  the famous shopping street",
    "  with electric trams",
])

make_image("bern_oldtown.png", 800, 500, "#DEB887", [
    "Altstadt Bern - UNESCO",
    "",
    "  Arcaded walkways (Lauben)",
    "  Zytglogge (Clock Tower)",
    "  Aare River flowing below",
    "  Medieval sandstone buildings",
])

make_image("tax_form_2024.png", 600, 800, "#ffffff", [
    "Einkommenssteuererklärung 2024",
    "Kanton Zürich",
    "═══════════════════════════════",
    "",
    "Nachname:     Müller",
    "Vorname:      Hans",
    "Strasse:      Bahnhofstrasse 42",
    "PLZ:          8001",
    "Ort:          Zürich",
    "AHV-Nummer:   756.1234.5678.90",
    "Zivilstand:   verheiratet",
    "Steuerjahr:   2024",
    "",
    "Einkünfte:",
    "  Bruttoeinkommen:   CHF 120'000",
    "  Abzüge:            CHF 15'000",
    "  Reineinkommen:     CHF 105'000",
])

make_image("ahv_form.png", 600, 500, "#ffffff", [
    "AHV/IV Anmeldung",
    "═══════════════════════════════",
    "",
    "AHV-Nummer:   756.9876.5432.10",
    "Name:         Schneider",
    "Vorname:      Maria",
    "Geburtsdatum: 15.03.1985",
    "Nationalität: Schweiz",
    "",
    "Eintrittsdatum: 01.04.2024",
])

make_image("residence_permit.png", 600, 500, "#ffffff", [
    "Aufenthaltsbewilligung",
    "Bewilligung B / Permis B",
    "═══════════════════════════════",
    "",
    "Name:           Weber, Thomas",
    "Nationality:    German",
    "Permit Type:    B (EU/EFTA)",
    "Valid From:     01.01.2024",
    "Valid Until:    31.12.2025",
    "Canton:         Zürich",
])

make_image("swiss_parliament.png", 800, 500, "#4169E1", [
    "Bundeshaus Bern",
    "",
    "  Swiss Federal Parliament",
    "  during a session",
    "",
    "  Swiss flag flying above",
    "  the Bundeshaus dome",
])

make_image("swiss_train_accident.png", 800, 500, "#808080", [
    "SBB Zugunglück bei Luzern",
    "",
    "  Entgleister Zug",
    "  Rettungskräfte vor Ort",
    "  Einsatzfahrzeuge am Gleis",
])

make_image("swiss_alps_event.png", 800, 500, "#FFFFFF", [
    "Ski Alpin - Weltcup",
    "",
    "  Rennstrecke in den Alpen",
    "  Schweizer Fahnen",
    "  Zuschauer an der Piste",
    "  Athleten im Finish-Bereich",
])

print(f"\nDone. {len(os.listdir(IMAGES_DIR))} images in {IMAGES_DIR}")
