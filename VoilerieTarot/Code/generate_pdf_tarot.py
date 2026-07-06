#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:44:28 2026

@author: jadeleroux
"""
import io, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image, HRFlowable, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas


# ── Styles Visuels ────────────────────────────────────────────────────────────
def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("TitreLayout", parent=s["Normal"], fontSize=15,
                         alignment=TA_CENTER, fontName="Helvetica-Bold", leading=18))
    s.add(ParagraphStyle("ColHeader", fontSize=9.5, leading=12, fontName="Helvetica"))
    s.add(ParagraphStyle("ColHeaderR", fontSize=9.5, leading=12, alignment=TA_RIGHT, fontName="Helvetica"))
    s.add(ParagraphStyle("LabelAngle", fontSize=12, fontName="Helvetica-Bold", leftIndent=1*cm))
    s.add(ParagraphStyle("SectionTitle", parent=s["Normal"], fontSize=11, fontName="Helvetica-Bold",
                         textColor=colors.black, spaceBefore=12, spaceAfter=6))
    s.add(ParagraphStyle("CellText", parent=s["Normal"], fontSize=9, fontName="Helvetica"))
    s.add(ParagraphStyle("CellLabel", parent=s["Normal"], fontSize=9, fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("NoteStyle", parent=s["Normal"], fontSize=10, fontName="Courier",
                         textColor=colors.black, leftIndent=0.5*cm, spaceBefore=4))
    return s


# ── Numérotation des pages ───────────────────────────────────────────────────
class NumeroPageCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 8)
        self.drawCentredString(A4[0]/2, 0.8*cm, f"{self._pageNumber} / {page_count}")


def _fig2img(fig, w_max_cm=18.5, h_max_cm=19.0, pad=0.1):
    buf = io.BytesIO()
    # On explose le DPI à 600 (Qualité impression pro)
    # On ajoute antialiased pour lisser les traits de la voile
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", 
                pad_inches=pad, facecolor="white", transparent=False)
    buf.seek(0)
    
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    w = w_max_cm * cm
    h = w * ratio
    if h > h_max_cm * cm:
        h = h_max_cm * cm
        w = h / ratio
        
    img.drawWidth = w
    img.drawHeight = h
    img.hAlign = 'CENTER'
    return img


# ── Dessins Techniques (Laizes & Renforts) ──────────────────────────────────
def _draw_patch_detail(coords_fab, courbures, bords, nb_pts, idx_corner, labels, renforts_cm, start_num):
    pts = np.array(coords_fab); px, py = pts[idx_corner]; gx, gy = np.mean(pts[:nb_pts], axis=0)
    fig, ax = plt.subplots(figsize=(3.5, 2.5)); ax.set_aspect("equal")
    rayons = renforts_cm.get(labels[idx_corner], [30.0]); rmax = rayons[-1]; limit = rmax + 10

    def get_cp(p1, p2, pct):
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        vx, vy = p2[0]-p1[0], p2[1]-p1[1]; d = math.sqrt(vx**2+vy**2)
        if d == 0: return (mx, my)
        nx, ny = -vy/d, vx/d
        if nx*(gx-mx)+ny*(gy-my) < 0: nx, ny = -nx, -ny
        return (mx+nx*(pct/100)*d*2, my+ny*(pct/100)*d*2)

    p_prev = pts[(idx_corner-1)%nb_pts]; cp_prev = get_cp(p_prev, pts[idx_corner], courbures[bords[(idx_corner-1)%nb_pts]])
    p_next = pts[(idx_corner+1)%nb_pts]; cp_next = get_cp(pts[idx_corner], p_next, courbures[bords[idx_corner]])
    verts_mask = [pts[idx_corner], cp_next, p_next, (gx, gy), p_prev, cp_prev, pts[idx_corner]]
    codes_mask = [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE3]
    corner_path = mpath.Path(verts_mask, codes_mask); patch_mask = mpatches.PathPatch(corner_path, fc="none", ec="none"); ax.add_patch(patch_mask)
    t = np.linspace(0, 1, 50)
    for p_a, cp_a, p_b in [(p_prev, cp_prev, pts[idx_corner]), (pts[idx_corner], cp_next, p_next)]:
        ax.plot((1-t)**2*p_a[0] + 2*(1-t)*t*cp_a[0] + t**2*p_b[0],
                (1-t)**2*p_a[1] + 2*(1-t)*t*cp_a[1] + t**2*p_b[1],
                color="black", lw=1.6, zorder=10)

    v1 = (p_prev - pts[idx_corner]) / np.linalg.norm(p_prev - pts[idx_corner])
    v2 = (p_next - pts[idx_corner]) / np.linalg.norm(p_next - pts[idx_corner])
    bis = (v1 + v2) / np.linalg.norm(v1 + v2)

    current_n = start_num
    DECAL_DOUBLON = 6   # cm de décalage radial par doublon
    for j, r in enumerate(rayons):
        ax.add_patch(plt.Circle((px, py), r, color="black", fill=False, lw=0.6,
                                zorder=5, clip_path=patch_mask))
        idx_doublons = [k for k, rk in enumerate(rayons) if abs(rk - r) < 0.1]
        pos = idx_doublons.index(j)
        r_lab = max(r - 6 - pos * DECAL_DOUBLON, 3.0)
        tx = px + bis[0] * r_lab
        ty = py + bis[1] * r_lab
        ax.text(tx, ty, str(current_n), fontsize=7, ha='center', va='center',
                fontweight='bold', color="black")
        current_n += 1
    ax.set_xlim(px-limit, px+limit); ax.set_ylim(py-limit, py+limit); ax.axis("off")
    return fig, current_n


def _draw_sail(coords_fab, courbures, bords, nb_pts, labels, idx_cote_laize,
               list_pas, renforts_cm, config_angles, codes_accastillage=None, fixation_angles=None):
    pts = np.array(coords_fab); gx, gy = np.mean(pts[:nb_pts], axis=0)
    fig, ax = plt.subplots(figsize=(10, 11)); ax.set_aspect("equal")

    def get_cp(p1, p2, pct):
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        vx, vy = p2[0]-p1[0], p2[1]-p1[1]; d = math.sqrt(vx**2+vy**2)
        if d == 0: return (mx, my)
        nx, ny = -vy/d, vx/d
        if nx*(gx-mx)+ny*(gy-my) < 0: nx, ny = -nx, -ny
        return (mx+nx*(pct/100)*d*2, my+ny*(pct/100)*d*2)

    # 1. CONSTRUCTION DU CONTOUR
    verts, codes, pts_contour = [], [], []
    for i in range(nb_pts):
        p1, p2 = pts[i], pts[(i+1)%nb_pts]; cp = get_cp(p1, p2, courbures[bords[i]])
        if i == 0: verts.append(p1); codes.append(mpath.Path.MOVETO)
        verts.extend([cp, p2]); codes.extend([mpath.Path.CURVE3, mpath.Path.CURVE3])
        for t in np.linspace(0, 1, 50):
            pts_contour.append(((1-t)**2*p1[0]+2*(1-t)*t*cp[0]+t**2*p2[0],
                                (1-t)**2*p1[1]+2*(1-t)*t*cp[1]+t**2*p2[1]))
    
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, fc="none", ec="black", lw=1.5, zorder=10)
    ax.add_patch(patch)

    # 2. CALCUL DES AXES DE LAIZES
    p1_ref, p2_ref = pts[idx_cote_laize], pts[(idx_cote_laize+1)%nb_pts]
    angle = math.atan2(p2_ref[1]-p1_ref[1], p2_ref[0]-p1_ref[0])
    nx_l, ny_l = -math.sin(angle), math.cos(angle); dx_l, dy_l = math.cos(angle), math.sin(angle)

    projs = [p[0]*nx_l + p[1]*ny_l for p in pts_contour]
    d_min, d_max = min(projs), max(projs)

    # 3. TRACÉ DES LIGNES ET NUMÉROS (Simplifié car list_pas provient de l'UI en cm !)
    span = (max(pts[:,0])-min(pts[:,0]) + max(pts[:,1])-min(pts[:,1]))*2
    cumul_dessin = 0.0
    
    # ATTENTION : Si list_pas contient les valeurs "W" (avec recouvrement), 
    # on retire le recouvrement pour le dessin des lignes intérieures :
    RECOUVR = 2.5 if any(b == "Tentmesh" for b in bords) else 2.5 # Ajuste si besoin
    
    for i, w_cm in enumerate(list_pas):
        # On repasse de la largeur brute à couper au pas physique du tracé
        pas = w_cm - RECOUVR if i < len(list_pas) - 1 else w_cm
        
        d_start = d_min + cumul_dessin
        d_end = d_min + cumul_dessin + pas
        cumul_dessin += pas

        if i < len(list_pas) - 1:
            lx, ly = d_end * nx_l, d_end * ny_l
            ax.plot([lx - dx_l*span, lx + dx_l*span],
                    [ly - dy_l*span, ly + dy_l*span],
                    color="black", lw=0.6, zorder=5, clip_path=patch)

        pts_dans_panneau = [p for p in pts_contour if d_start <= (p[0]*nx_l + p[1]*ny_l) <= d_end]
        if pts_dans_panneau:
            tx, ty = np.mean(pts_dans_panneau, axis=0)
        else:
            tx, ty = (d_start + pas/2)*nx_l, (d_start + pas/2)*ny_l

        ax.text(tx, ty, str(i + 1), fontsize=10, ha='center', va='center',
                fontweight='bold', color="black", zorder=20,
                bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='none', alpha=0.8))

    # 4. ÉTIQUETTES DES ANGLES
    for i in range(nb_pts):
        px, py = pts[i]; pt_name = labels[i]
        ret = config_angles.get(pt_name, 0)
        vx, vy = px-gx, py-gy
        dist = math.sqrt(vx**2+vy**2); ux, uy = vx/dist, vy/dist
        
        if pt_name in renforts_cm:
            for r in renforts_cm[pt_name]:
                ax.add_patch(plt.Circle((px, py), r, color="black", fill=False, lw=0.6, zorder=6, clip_path=patch))

        code = codes_accastillage.get(pt_name, "VOMC") if codes_accastillage else "VOMC"
        fixation = fixation_angles.get(pt_name, "") if fixation_angles else ""
        label_text = f"{pt_name}\n{code} - {ret}cm" + (f"\n{fixation}" if fixation else "")
        ax.text(px + ux * 75, py + uy * 75, label_text, fontsize=10, fontweight='bold', 
                ha='center', va='center', ma='center', color="black", 
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.8, alpha=0.9))

    # Échelle de plan
    x_min, x_max, y_min = min(pts[:,0]), max(pts[:,0]), min(pts[:,1])
    scale_length = 200.0  
    scale_x_start = ((x_min + x_max) / 2) - (scale_length / 2)  
    scale_y = y_min - 120  
    
    ax.plot([scale_x_start, scale_x_start + scale_length], [scale_y, scale_y], color='black', lw=2.0, zorder=30, clip_on=False)
    ax.plot([scale_x_start, scale_x_start], [scale_y - 4, scale_y + 4], color='black', lw=1.2, zorder=30, clip_on=False)
    ax.plot([scale_x_start + scale_length, scale_x_start + scale_length], [scale_y - 4, scale_y + 4], color='black', lw=1.2, zorder=30, clip_on=False)
    ax.text(scale_x_start + scale_length/2, scale_y + 8, "2 m", fontsize=9, fontweight='bold', ha='center', va='bottom', color='black')
    ax.set_xlim(min(pts[:,0]) - 80, max(pts[:,0]) + 80)
    ax.set_ylim(min(pts[:,1]) - 150, max(pts[:,1]) + 80)
    ax.axis("off")
    ax.patch.set_visible(False)
    fig.patch.set_visible(False)
    return fig


def calculer_L_max_draille(coords, indices_draille, nb_pts):
    """
    Calcule la plus grande perpendiculaire vers la draille (bord ou diagonale).
    """
    if not indices_draille or len(indices_draille) != 2:
        return 0.0
    idx1, idx2 = list(indices_draille)
    p1 = np.array(coords[idx1])
    p2 = np.array(coords[idx2])
    vec_draille = p2 - p1
    norm_draille = np.linalg.norm(vec_draille)
    if norm_draille == 0: return 0.0
    distances = []
    for i in range(nb_pts):
        if i not in indices_draille:
            pP = np.array(coords[i])
            dist = abs(np.cross(p2 - p1, p1 - pP)) / norm_draille
            distances.append(dist)
    return max(distances) / 100 if distances else 0.0


# ── Fonction Principale d'Export ──────────────────────────────────────────────
def generer_pdf(nom_client, tissu, coloris, date_str, type_voile,
                surf_m2, aire_pleine, laize_choisie, largeur_dispo, laize_a_couper, reste_final,
                nb_laizes, formule_bdf, nb_bandes_rouleau, largeur_bdf_choisie,
                laize_utile_cm, strategie, renforts_cm, labels, longueurs_arcs_metres,
                perte=None, long_rouleau_cm=None, coords_fab=None, courbures=None,
                bords=None, nb_pts=4, config_angles=None, pts_emm=None,
                l_max=0.0, codes_accastillage=None, reference="", list_pas=None, idx_cote_laize=0, 
                debit_total_atelier=0.0, poids_surfacique=0, L_max_lin=0.0, longueur_bdf_cm=0, largeurs_bdf=None, longueurs_bdf=None, fixation_angles=None, list_longueurs_brutes=None):
    if fixation_angles is None:
        fixation_angles = {}
    S = _styles(); buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1*cm, rightMargin=1*cm,
                             topMargin=0.3*cm, bottomMargin=1*cm)
    story = []

    def create_header():
        logo_path = "/Users/jadeleroux/Documents/Polytech/Stage 4A/Voilerie Tarot/Logo/logo_voilerie.png"
        logo_img = Image(logo_path, width=1.8*cm, height=1.8*cm) if os.path.exists(logo_path) else ""
        type_voile_complet = type_voile
        if pts_emm:
            type_voile_complet += " sur enrouleur"
        col1 = [Paragraph(f"Date: {date_str}", S["ColHeader"]),
                Paragraph("<b>Voilerie Tarot</b>", S["ColHeader"]),
                Paragraph(f"Référence : <b>{reference or '—'}</b>")]
        col2 = [Paragraph(f"Client: <b>{nom_client or '—'}</b>", S["ColHeaderR"]),
                Paragraph(f"Type voile : <b>{type_voile_complet}</b>", S["ColHeaderR"]),
                Paragraph(f"Matériau : {tissu}", S["ColHeaderR"]),
                Paragraph(f"Coloris : {coloris}", S["ColHeaderR"])]
        h_table = Table([[logo_img, col1, col2]], colWidths=[2.2*cm, 8.4*cm, 8.4*cm])
        h_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                                     ('LEFTPADDING', (0,0), (-1,-1), 0),
                                     ('TOPPADDING', (0,0), (-1,-1), 0)]))
        return h_table

    # ── PAGE 1 ──
    story.append(create_header())
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=3))
    story.append(Paragraph("SCHÉMA DE LAIZE", S["TitreLayout"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceBefore=3))
    story.append(Spacer(1, 0.4*cm))
    if coords_fab:
        # On utilise directement l'index passé en argument (celui de la visu)
        fig1 = _draw_sail(coords_fab, courbures, bords, nb_pts, labels, idx_cote_laize,
                          list_pas, renforts_cm, config_angles,
                          codes_accastillage=codes_accastillage, fixation_angles=fixation_angles)
        story.append(_fig2img(fig1, h_max_cm=19.5)); plt.close(fig1)
    story.append(PageBreak())

    # ── PAGE 2 ──
    story.append(create_header())
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=3))
    story.append(Paragraph("SCHÉMA DES RENFORTS", S["TitreLayout"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceBefore=3))
    story.append(Spacer(1, 0.6*cm))
    if coords_fab:
        cur_n = int(nb_laizes) + 1
        for i in range(nb_pts):
            fig_p, cur_n = _draw_patch_detail(coords_fab, courbures, bords, nb_pts, i,
                                              labels, renforts_cm, cur_n)
            story.append(_fig2img(fig_p, w_max_cm=10, h_max_cm=5.5, pad=0.02)); plt.close(fig_p)
            story.append(Paragraph(f"<b>Angle {labels[i]}</b>", S["LabelAngle"]))
            story.append(Spacer(1, 0.3*cm))
            story.append(HRFlowable(width="90%", thickness=0.3, color=colors.black,
                                    spaceBefore=5, spaceAfter=5))
    story.append(PageBreak())

    # ── PAGE 3 ──
    story.append(create_header())
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=3))
    story.append(Paragraph("FICHE TECHNIQUE", S["TitreLayout"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceBefore=3))

    # --- SECTION SURFACES ET MATÉRIAU ---
    story.append(Paragraph("Surfaces et Matériau", S["SectionTitle"]))
    mat_data = [
        [Paragraph("Matériau", S["CellLabel"]), Paragraph(f"{tissu.upper()}", S["CellText"]),
         Paragraph("Coloris", S["CellLabel"]), Paragraph(f"{coloris.upper() if coloris else '-'}", S["CellText"])]
    ]
    t_mat = Table(mat_data, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm])
    t_mat.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke)
    ]))
    story.append(t_mat)
    story.append(Spacer(1, 0.6*cm))

    # Calcul de la masse en kg à partir des valeurs de l'interface
    masse_totale_kg = (debit_total_atelier * poids_surfacique) / 1000

    surf_data = [
        [Paragraph("Surface réelle", S["CellLabel"]), Paragraph(f"{surf_m2:.2f} m²", S["CellText"]),
         Paragraph("Laize rouleau", S["CellLabel"]), Paragraph(f"{laize_choisie} cm", S["CellText"])],
        [Paragraph("Surface matériau", S["CellLabel"]), Paragraph(f"{debit_total_atelier:.2f} m²", S["CellText"]),
         Paragraph("Largeur disponible", S["CellLabel"]), Paragraph(f"{largeur_dispo:.1f} cm", S["CellText"])],
        [Paragraph("Longueur Rouleau", S["CellLabel"]), Paragraph(f"{long_rouleau_cm/100:.0f} ml", S["CellText"]),
         Paragraph("Poids total estimé", S["CellLabel"]), Paragraph(f"{masse_totale_kg:.2f} kg" , S["CellText"])],
    ]
    t_surf = Table(surf_data, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm])
    t_surf.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke)
    ]))
    story.append(t_surf)

    # --- SECTION DÉTAIL DE LA COUPE ---
    story.append(Paragraph("Détail de la coupe", S["SectionTitle"]))

    # Table Détail de la coupe (commune aux deux modes)
    cote_ref = f"Bord {labels[idx_cote_laize]}{labels[(idx_cote_laize+1)%nb_pts]}"
    coupe_data = [
        [Paragraph("Nb de laizes", S["CellLabel"]), Paragraph(str(nb_laizes), S["CellText"]),
         Paragraph("L_max", S["CellLabel"]), Paragraph(f"{L_max_lin:.2f} m", S["CellText"])],
        [Paragraph("Côté de référence", S["CellLabel"]), Paragraph(cote_ref, S["CellText"]),
         Paragraph("Hauteur de coupe", S["CellLabel"]), Paragraph(f"{l_max*100:.1f} cm", S["CellText"])]
    ]
    t_coupe = Table(coupe_data, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm])
    t_coupe.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke)
    ]))
    story.append(t_coupe)

    # Détail des largeurs 
    if list_pas:
        story.append(Spacer(1, 0.3*cm))
        detail_data = []
        temp_row = []
        for i, largeur in enumerate(list_pas):
            temp_row.extend([Paragraph(f"Laize #{i+1}", S["CellLabel"]),
                             Paragraph(f"{largeur:.1f} cm", S["CellText"])])
            if len(temp_row) == 4:
                detail_data.append(temp_row)
                temp_row = []
        if temp_row:
            temp_row.extend([Paragraph("", S["CellLabel"]), Paragraph("", S["CellText"])])
            detail_data.append(temp_row)
        t_det = Table(detail_data, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm])
        t_det.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke)
        ]))
        story.append(t_det)

    # Longueurs prises de côtes
    if list_longueurs_brutes:
        story.append(Paragraph("Longueurs brutes", S["SectionTitle"]))
        brut_data = []
        temp_row = []
        items = list(list_longueurs_brutes.items())
        for i, (b, val_cm) in enumerate(items):
            temp_row.extend([Paragraph(f"Bord {b}", S["CellLabel"]),
                             Paragraph(f"{val_cm/100:.3f} m", S["CellText"])])
            if len(temp_row) == 4:
                brut_data.append(temp_row)
                temp_row = []
        if temp_row:
            temp_row.extend([Paragraph("", S["CellLabel"]), Paragraph("", S["CellText"])])
            brut_data.append(temp_row)
        t_brut = Table(brut_data, colWidths=[3*cm, 6*cm, 3*cm, 6*cm])
        t_brut.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke)
        ]))
        story.append(t_brut)

    
    # --- SECTION LONGUEURS DE COUPE ---
    story.append(Paragraph("Longueurs côtés", S["SectionTitle"]))
    
    # Séparation des bords standards et des diagonales
    bords_data_dict = {}
    diagonales_data_dict = {}
    
    # 1. Tri des données reçues
    for b, d in longueurs_arcs_metres.items():
        nom_propre = b.replace("'", "").strip()
        if len(nom_propre) == 2:
            try:
                idx0 = labels.index(nom_propre[0])
                idx1 = labels.index(nom_propre[1])
                diff = abs(idx0 - idx1)
                if diff == 1 or diff == (len(labels) - 1):
                    bords_data_dict[b] = d
                else:
                    diagonales_data_dict[b] = d
            except ValueError:
                bords_data_dict[b] = d
        else:
            bords_data_dict[b] = d

    # Force la présence des deux diagonales théoriques (ex: AC et BD) si elles manquent
    # pour éviter que le tableau disparaisse
    if len(labels) == 4:
        diag1_cle = f"{labels[0]}{labels[2]}"  # AC
        diag2_cle = f"{labels[1]}{labels[3]}"  # BD
        if diag1_cle not in diagonales_data_dict and f"{labels[2]}{labels[0]}" not in diagonales_data_dict:
            diagonales_data_dict[diag1_cle] = None
        if diag2_cle not in diagonales_data_dict and f"{labels[3]}{labels[1]}" not in diagonales_data_dict:
            diagonales_data_dict[diag2_cle] = None

    # 2. Remplissage du tableau des Bords Extérieurs
    arcs_data = []
    temp_row = []
    for i, (b, d) in enumerate(bords_data_dict.items()):
        temp_row.extend([Paragraph(f"<b>Bord {b}</b>", S["CellLabel"]),
                         Paragraph(f"{d:.3f} m", S["CellText"])])
        if (i+1) % 2 == 0 or (i+1) == len(bords_data_dict):
            while len(temp_row) < 4:
                temp_row.extend([Paragraph("", S["CellLabel"]), Paragraph("", S["CellText"])])
            arcs_data.append(temp_row)
            temp_row = []
            
    t_arcs = Table(arcs_data, colWidths=[3*cm, 6*cm, 3*cm, 6*cm])
    t_arcs.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke)
    ]))
    story.append(t_arcs)

    # 3. Remplissage du tableau des Diagonales (toujours visible si 4 angles)
    if diagonales_data_dict:
        story.append(Spacer(1, 0.2*cm))
        diag_data = []
        temp_row_diag = []
        for i, (b, d) in enumerate(diagonales_data_dict.items()):
            val_txt = f"{d:.3f} m" if d is not None else "— m"
            temp_row_diag.extend([Paragraph(f"<b>Diag. {b}</b>", S["CellLabel"]),
                                  Paragraph(val_txt, S["CellText"])])
            if (i+1) % 2 == 0 or (i+1) == len(diagonales_data_dict):
                while len(temp_row_diag) < 4:
                    temp_row_diag.extend([Paragraph("", S["CellLabel"]), Paragraph("", S["CellText"])])
                diag_data.append(temp_row_diag)
                temp_row_diag = []
                
        t_diag = Table(diag_data, colWidths=[3*cm, 6*cm, 3*cm, 6*cm])
        t_diag.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke)
        ]))
        story.append(t_diag)

    # ── Notes TECHNIQUES ──
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("NOTES TECHNIQUES", S["SectionTitle"]))
    story.append(HRFlowable(width="100%", thickness=0.3, color=colors.black))
    story.append(Paragraph(f"STRATÉGIE DE COUPE : {strategie.upper()}", S["NoteStyle"]))
    if tissu.lower() != "tentmesh":
        story.append(Paragraph("BANDES DROIT FIL :", S["NoteStyle"]))
    if largeurs_bdf:
        for larg, long_m in zip(largeurs_bdf, longueurs_bdf or [0.0] * len(largeurs_bdf)):
            val = f"{long_m:.1f} ml" if long_m > 0 else "longueur auto"
            story.append(Paragraph(f"   • 1 bande de {larg:.0f} cm de {val}", S["NoteStyle"]))
    else:
        story.append(Paragraph(f"   {formule_bdf.upper()}", S["NoteStyle"]))
    story.append(PageBreak())

    # ── PAGE 4 ──
    story.append(create_header())
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=3))
    story.append(Paragraph("NOMENCLATURE", S["TitreLayout"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceBefore=3))
    story.append(Paragraph("Accastillage", S["SectionTitle"]))
    
    
    points_emm = list(pts_emm) if isinstance(pts_emm, (list, set, range)) else []
    MATERIEL_REF = {
        "VOP1": ["1 poulie 96.30", "{L}m de Top Cruising 8mm blanc", "1 manille automatique 28.45"],
        "VOP2": ["1 poulie 96.40 <i>(à ringot et taquet coinceur)</i>", "1 poulie 96.30",
                 "{L}m de Top Cruising 8mm blanc", "1 manille automatique 28.45", "1 manille lyre 28.13"],
        "VOP3": ["1 poulie 96.52 <i>(à ringot et taquet coinceur)</i>", "1 poulie 96.31",
                 "{L}m de Top Cruising 8mm blanc", "1 manille automatique 28.45", "1 manille lyre 28.13"],
        "VOPE": ["1 poulie 9.65 <i>(à ringot et coinceur fente)</i>", "1 poulie 9.64",
                 "4,40m de Top Cruising 5mm blanc", "1 manille automatique 28.45", "1 manille lyre 28.13"],
        "VOGE": ["1 emmagasineur FXS90 ou FXS150", "1 kit de cordage pour emmagasineur", "1 clé Allen"],
        "VOMM": ["1 mousqueton pompier 29.21", "1 manille torse 28.25"],
        "VOMC": ["Tresse dyneema 3mm", "1 mousqueton pompier 29.21"]
    }
    
    # --- 1. IDENTIFICATION DES EMMAGASINEURS (VOPE/VOGE) ---
    indices_emmag = []
    for i, pt in enumerate(labels):
        code_complet = codes_accastillage.get(pt, "")
        if code_complet.startswith("VOPE") or code_complet.startswith("VOGE"):
            indices_emmag.append(i)

    # --- 2. PRÉPARATION DES DONNÉES DU TABLEAU ---
    nomenc_data = [[Paragraph("<b>Angle</b>", S["CellLabel"]),
                    Paragraph("<b>Type</b>", S["CellLabel"]),
                    Paragraph("<b>Accroche</b>", S["CellLabel"]),
                    Paragraph("<b>Détail Matériel</b>", S["CellLabel"])]]

    # --- 3. BOUCLE DE REMPLISSAGE ---
    for i, pt in enumerate(labels):
        # A. Logique d'accroche (Sangles / Anneau sanglé / Oeillet)
        if len(indices_emmag) > 0:
            accroche_auto = "Sangles" if i in indices_emmag else "Anneau sanglé"
        else:
            if surf_m2 > 10.0:
                accroche_auto = "Triangle"
            else:
                accroche_auto = "Oeillet"

        # B. Récupération et filtrage du matériel
        code_complet = codes_accastillage.get(pt, "VOMC")
        code_racine = code_complet.split(" - ")[0] 
        items = MATERIEL_REF.get(code_racine, ["Matériel standard"])
        
        # Filtre Manille Lyre 28.13 si voile triangulaire
        if nb_pts == 3:
            items = [ligne for ligne in items if "28.13" not in ligne]
            
        if accroche_auto == "Triangle":
            items = [ligne for ligne in items if "28.13" not in ligne]

        # C. Calcul des longueurs de cordage
        val_l_numerique = 0.0
        if (i not in indices_emmag) and l_max > 0:
            base_calc = l_max + 1.40 + 0.35
            if code_racine == "VOP1":   val_l_numerique = base_calc + 1.0
            elif code_racine == "VOP2": val_l_numerique = 2 * base_calc + 1.0
            elif code_racine == "VOP3": val_l_numerique = 3 * base_calc + 1.0
            elif code_racine == "VOPE": val_l_numerique = base_calc + 1.0
        else:
            classiques = {"VOP1": 2.25, "VOP2": 4.00, "VOP3": 5.75}
            val_l_numerique = classiques.get(code_racine, 0.0)

        txt_final = f"{val_l_numerique:.2f}m" if val_l_numerique > 0 else ""

        # D. Formatage des lignes de matériel (puces)
        materiel_pour_tableau = []
        for ligne in items:
            nouvelle_ligne = ligne.replace("{L}m", txt_final).replace("{L}", txt_final)
            materiel_pour_tableau.append(Paragraph(f"• {nouvelle_ligne}", S["CellText"]))

        # E. Ajout de la ligne au tableau
        nomenc_data.append([
            Paragraph(f"<b>{pt}'</b>", S["CellText"]),
            Paragraph(code_racine, S["CellText"]),
            Paragraph(accroche_auto, S["CellText"]),
            materiel_pour_tableau
        ])

    # --- 4. CRÉATION FINALE DU TABLEAU ---
    t_nomenc = Table(nomenc_data, colWidths=[2.0*cm, 2.0*cm, 3.0*cm, 10.0*cm])
    t_nomenc.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(t_nomenc)

    doc.build(story, canvasmaker=NumeroPageCanvas)
    buf.seek(0)
    return buf.read()