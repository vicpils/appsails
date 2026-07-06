#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
from datetime import date
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# =============================================================================
# 1. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Voilerie Tarot - Calculateur VO", layout="wide")
# =============================================================================
# 2. RÉFÉRENTIEL TECHNIQUE
# =============================================================================
MATERIAUX = {
    "Soltis 92": {
        "poids": 420, 
        "laizes": {177: 50, 267: 40},
        "rupture_ch": 310, "rupture_tr": 210, # daN/5cm
        "dechirure_ch": 45, "dechirure_tr": 20  # daN
    },
    "Soltis 86": {
        "poids": 380, 
        "laizes": {177: 50, 267: 40},
        "rupture_ch": 230, "rupture_tr": 160, # daN/5cm
        "dechirure_ch": 45, "dechirure_tr": 20  # daN
    },
    "Tentmesh": {
        "poids": 340, 
        "laizes": {300: 50},
        "rupture_ch": 140, "rupture_tr": 230, 
        "dechirure_ch": 40, "dechirure_tr": 55
    },
    "Meaban": {
        "poids": 500, 
        "laizes": {300: 30},
        "rupture_ch": 260, "rupture_tr": 200,
        "dechirure_ch": 40, "dechirure_tr": 30
    },
    "Autre...": {"poids": 0, "laizes": {}}
}
NUANCIERS_PDF = {
    "Soltis 92" : "SOLTIS_92_nuancier.pdf",
    "Soltis 86" : "SOLTIS_86_nuancier.pdf",
    "Tentmesh"  : "TENTMESH_nuancier.pdf",
    "Meaban"    : "MEABAN_nuancier.pdf",
}
DOSSIER_NUANCIERS  = "/Users/jadeleroux/Documents/Polytech/Stage 4A/Voilerie Tarot/Nuanciers"
RECOUVR_STD        = 2.5
RECOUVR_TENT       = 3.0
OFFSET_BORDURE     = 7.0
COEFF_BDF          = 1.05
LABELS             = "ABCDE"
COULEURS_RENFORT   = ['#e67e22','#8e44ad','#27ae60','#c0392b','#2980b9']
IDX_DIAG           = {"BD": (1,3), "AC": (0,2), "AD": (0,3)}
BORDS_PAR_NB       = {3: ["AB","BC","CA"], 4: ["AB","BC","CD","DA"], 5: ["AB","BC","CD","DE","EA"]}
ACCASTILLAGE = {
    "VOMC - 12cm": 12, "VOMM - 8cm": 8,
    "VOP1 - 30cm": 30, "VOP2 - 35cm": 35, "VOP3 - 40cm": 40,
    "VOPE - 50cm": 50, "VOGE - 12cm": 12, "Autre": 0,
}
ACCASTILLAGE_EMM = {"VOPE - 50cm", "VOGE - 12cm"}

# =============================================================================
# 3. FONCTIONS MATHÉMATIQUES
# =============================================================================
def aire_triangle(a, b, c):
    if a + b <= c or a + c <= b or b + c <= a: return 0
    s = (a + b + c) / 2
    return math.sqrt(max(0, s*(s-a)*(s-b)*(s-c)))
def generer_coords_triangle(la, lb, lc):
    if la+lb <= lc or la+lc <= lb or lb+lc <= la: return None
    xc = (la**2 + lc**2 - lb**2) / (2*la)
    return [(0.0, 0.0), (xc, math.sqrt(max(0, lc**2 - xc**2))), (la, 0.0)]
def generer_coords_quadrilatere(ab, bc, cd, da, bd):
    if any(x <= 0 for x in [ab, bc, cd, da, bd]): return None
    try:
        yb = (bd**2 - ab**2 + da**2) / (2*da)
        xb = math.sqrt(max(0, bd**2 - yb**2))
        ux, uy = xb/bd, yb/bd
        proj = (cd**2 - bc**2 + bd**2) / (2*bd)
        h = math.sqrt(max(0, cd**2 - proj**2))
        pC = (proj*ux + h*uy, proj*uy - h*ux)
        return [(0.0, float(da)), (xb, yb), pC, (0.0, 0.0)]
    except Exception: return None
def generer_coords_pentagone(ab, bc, cd, de, ea, ac, ad):
    coords = generer_coords_quadrilatere(ab, bc, cd, ad, ac)
    if not coords: return None
    pA, pB, pC, pD = coords
    try:
        dist_ad = math.sqrt((pD[0]-pA[0])**2 + (pD[1]-pA[1])**2)
        if dist_ad == 0: return None
        ux, uy = (pD[0]-pA[0])/dist_ad, (pD[1]-pA[1])/dist_ad
        vx, vy = -uy, ux
        xe = (ea**2 + dist_ad**2 - de**2) / (2*dist_ad)
        ye = math.sqrt(max(0, ea**2 - xe**2))
        if (pC[0]-pA[0])*vx + (pC[1]-pA[1])*vy > 0: vx, vy = -vx, -vy
        return [pA, pB, pC, pD, (pA[0]+xe*ux+ye*vx, pA[1]+xe*uy+ye*vy)]
    except Exception: return None
def longueur_cote(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
def calculer_longueur_arc(corde, pct):
    if corde <= 0: return 0
    f = (pct/100) * corde
    return corde * (1 + (8/3) * (f/corde)**2)
def fleche_bezier(p1, p2, pct, gx, gy):
    mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    vx, vy = p2[0]-p1[0], p2[1]-p1[1]
    dist = math.sqrt(vx**2+vy**2)
    if dist == 0: return (mx, my), 0
    nx, ny = -vy/dist, vx/dist
    if nx*(gx-mx)+ny*(gy-my) < 0: nx, ny = -nx, -ny
    f = (pct/100)*dist
    return (mx+nx*f*2, my+ny*f*2), f
def bezier_sub_to_radius(p_start, cp, p_end, rayon, n=128):
    """Bézier quadratique depuis p_start, tronquée à la distance rayon."""
    pts = [p_start]
    px, py, pd = p_start[0], p_start[1], 0.0
    for k in range(1, n + 1):
        t = k / n
        x = (1-t)**2*p_start[0] + 2*(1-t)*t*cp[0] + t**2*p_end[0]
        y = (1-t)**2*p_start[1] + 2*(1-t)*t*cp[1] + t**2*p_end[1]
        d = math.sqrt((x-p_start[0])**2 + (y-p_start[1])**2)
        if d >= rayon:
            alpha = (rayon - pd) / (d - pd) if d > pd else 0
            pts.append((px + alpha*(x-px), py + alpha*(y-py)))
            return pts
        pts.append((x, y))
        px, py, pd = x, y, d
    return pts

def calculer_aire_renfort(p_curr, cp_in, cp_out, p_prev, p_next, rayon, n=64):
    """Aire exacte d'un renfort (formule du lacet sur contour réel)."""
    cx, cy = p_curr
    side1 = bezier_sub_to_radius(p_curr, cp_in,  p_prev, rayon, n=n)
    side2 = bezier_sub_to_radius(p_curr, cp_out, p_next, rayon, n=n)
    end1, end2 = side1[-1], side2[-1]
    a1 = math.atan2(end1[1]-cy, end1[0]-cx)
    a2 = math.atan2(end2[1]-cy, end2[0]-cx)
    span = (a2 - a1) % (2*math.pi)
    if span > math.pi:
        span -= 2*math.pi
    arc_pts = [
        (cx + rayon*math.cos(a1 + span*t/32),
         cy + rayon*math.sin(a1 + span*t/32))
        for t in range(33)
    ]
    contour = side1 + arc_pts + list(reversed(side2))
    area = 0.0
    for j in range(len(contour)):
        x1, y1 = contour[j]
        x2, y2 = contour[(j+1) % len(contour)]
        area += x1*y2 - x2*y1
    return abs(area) / 2
# =============================================================================
# 4. FONCTIONS RENFORTS
# =============================================================================
def angle_entre(pA, pB, pC):
    vx1, vy1 = pA[0]-pB[0], pA[1]-pB[1]
    vx2, vy2 = pC[0]-pB[0], pC[1]-pB[1]
    d1, d2 = math.sqrt(vx1**2+vy1**2), math.sqrt(vx2**2+vy2**2)
    if d1 == 0 or d2 == 0: return 90.0
    return math.degrees(math.acos(max(-1, min(1, (vx1*vx2+vy1*vy2)/(d1*d2)))))
def obtenir_params_renforts(angle_deg, longueur_cote_cm):
    """Coefficient dégressive : 15% à 45°, 10% à 95°. Minimum 15cm ou 10% du côté."""
    if angle_deg <= 45:   coeff = 0.17
    elif angle_deg >= 95: coeff = 0.13
    else: coeff = 0.15 - (angle_deg - 45) * (0.05 / 50)
    valeur = longueur_cote_cm * coeff
    return max(min(valeur, 130.0), 20.0)
# Seuil au-delà duquel une voile est considérée comme "très grande"
# → renforts à 4 couches distinctes (au lieu de 3 avec la + petite doublée).
SEUIL_GRANDE_VOILE_M2 = 30.0
SEUIL_PETITE_VOILE_M2 = 10.0

def calculer_renforts(coords_fab, nb_pts, pts_emm=None, surf_m2=0.0, pts_full=None, N_SEG=None):
    """
    Retourne {'A': [r1, ..., rN], ...} — couches dégressives sur chaque angle.

    AFFICHAGE :
    - Angle avec emmagasineur           : 2 tailles
    - Angle normal, voile standard      : 3 tailles (la + petite sera doublée à la découpe)
    - Angle normal, très grande voile   : 4 tailles distinctes

    DÉCOUPE : `preparer_renforts_fabrication` duplique la + petite pour les angles
    à 3 couches (→ 4 pièces, les 2 + petites identiques).
    """
    if pts_emm is None:
        pts_emm = set()
    has_emm = len(pts_emm) >= 2
    grande_voile = surf_m2 > SEUIL_GRANDE_VOILE_M2
    res = {}
    for i in range(nb_pts):
        p_prev = coords_fab[(i-1) % nb_pts]
        p_curr = coords_fab[i]
        p_next = coords_fab[(i+1) % nb_pts]
        if pts_full is not None and N_SEG is not None:
            n_total = len(pts_full)
            gx = sum(p[0] for p in coords_fab) / nb_pts
            gy = sum(p[1] for p in coords_fab) / nb_pts
            def _u2(dx, dy):
                d = math.sqrt(dx**2+dy**2)
                return (dx/d, dy/d) if d > 1e-9 else (1.0, 0.0)
            idx_v    = i * N_SEG
            p_before = pts_full[(idx_v - 12) % n_total]
            p_after  = pts_full[(idx_v + 12) % n_total]
            tv1 = _u2(p_before[0]-p_curr[0], p_before[1]-p_curr[1])
            tv2 = _u2(p_after[0] -p_curr[0], p_after[1] -p_curr[1])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, tv1[0]*tv2[0]+tv1[1]*tv2[1]))))
        else:
            ang = angle_entre(p_prev, p_curr, p_next)
        l_ref  = (longueur_cote(p_prev, p_curr) + longueur_cote(p_curr, p_next)) / 2
        r_max  = obtenir_params_renforts(ang, l_ref)
        if i in pts_emm:
            # Emmagasineur : 2 couches (charge répartie par la drisse périphérique)
            r2 = r_max
            r1 = max(r2 / 1.2, 20.0)
            res[LABELS[i]] = sorted([round(r1, 1), round(r2, 1)])
        elif grande_voile:
            # Très grande voile (> 30 m²) : 4 tailles distinctes
            r4 = r_max
            r3 = max(r4 / 1.2, 20.0)
            r2 = max(r3 / 1.2, 20.0)
            r1 = max(r2 / 1.2, 20.0)
            res[LABELS[i]] = sorted([round(r1, 1), round(r2, 1), round(r3, 1), round(r4, 1)])
        elif surf_m2 < SEUIL_PETITE_VOILE_M2 and not (nb_pts == 3 and has_emm):
            r2 = r_max
            r1 = max(r2 / 1.2, 15.0)
            res[LABELS[i]] = sorted([round(r1, 1), round(r2, 1)])
        else:
            # Voile standard : 3 tailles → 4 pièces (les 2 + petites identiques)
            r3 = r_max
            r2 = max(r3 / 1.2, 15.0)
            r1 = max(r2 / 1.2, 15.0)
            res[LABELS[i]] = sorted([round(r1, 1), round(r2, 1), round(r3, 1)])
    return res
def preparer_renforts_fabrication(renforts_cm):
    """Si un angle a 3 couches, duplique la plus petite pour avoir 4 pièces à découper."""
    res = {}
    for pt, rayons in renforts_cm.items():
        if len(rayons) == 3:
            res[pt] = [rayons[0], rayons[0], rayons[1], rayons[2]]
        else:
            res[pt] = list(rayons)
    return res
# =============================================================================
# 5. TRACÉ
# =============================================================================
def tracer_voile(ax, coords_fab, coords_visu, coords_mur, courbures, bords, nb_pts,
                 laize_nette_cm, idx_cote_laize, renforts_cm,
                 pts_emm=None, grille=1.0, laize_theorique_cm=177,
                 mode_economique=False, nb_bandes_rouleau=1, largeur_bdf=7.0,
                 largeur_bdf_total=None, recouvr_actuel=2.5):
    if pts_emm is None: pts_emm = set()
    tous = coords_mur + coords_fab + coords_visu
    xs = [p[0] for p in tous]; ys = [p[1] for p in tous]
    s  = grille / max(max(xs)-min(xs), max(ys)-min(ys))
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    def norm(pts): return [((p[0]-cx)*s, (p[1]-cy)*s) for p in pts]
    p_mur = norm(coords_mur)
    p_fab = norm(coords_fab)
    gx = sum(p[0] for p in p_fab) / nb_pts
    gy = sum(p[1] for p in p_fab) / nb_pts
    # Grille et mur
    lim = grille/2 + grille*0.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.grid(True, lw=0.4, alpha=0.3, color='#aaaaaa', zorder=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)
    mx_p = [p[0] for p in p_mur]; my_p = [p[1] for p in p_mur]
    ax.fill(mx_p, my_p, color='#dddddd', alpha=0.25, zorder=1)
    ax.plot(mx_p+[mx_p[0]], my_p+[my_p[0]], color='#bbbbbb', lw=1.2, zorder=2)
    # Lignes de tension / pointillés emmagasineur
    for i, p in enumerate(p_mur):
        if i not in pts_emm:
            ax.plot([p[0], gx], [p[1], gy], color='#cccccc', ls='--', lw=1.0, zorder=2)
    emm_list = sorted(pts_emm)
    for k in range(len(emm_list)):
        for l in range(k+1, len(emm_list)):
            i, j = emm_list[k], emm_list[l]
            ax.plot([p_fab[i][0], p_fab[j][0]], [p_fab[i][1], p_fab[j][1]],
                    color='#aaaaaa', ls='--', lw=1.2, zorder=2)
    # G et labels mur / fabrication
    ax.plot(gx, gy, 'ko', ms=7, zorder=10)
    ax.text(gx, gy, "  G", fontsize=10, fontweight='bold', va='bottom', zorder=10)
    for i, p in enumerate(p_mur):
        ax.plot(p[0], p[1], 'o', color="#666666", ms=4, zorder=5)
        ax.text(p[0], p[1], f"  {LABELS[i]}", color="#666666", fontsize=9, zorder=6, alpha=0.8)
    for i, p in enumerate(p_fab):
        ax.plot(p[0], p[1], 'o', color="#1a6fbf", ms=5, zorder=12)
        ax.text(p[0], p[1], f"  {LABELS[i]}'", color="#1a6fbf", fontweight='bold', fontsize=11, zorder=13)
    # Voile courbée — chemin Bézier commun rouge + bleu
    path_data = []
    for i in range(nb_pts):
        p1, p2 = p_fab[i], p_fab[(i+1)%nb_pts]
        cp, _ = fleche_bezier(p1, p2, courbures[bords[i]], gx, gy)
        if i == 0: path_data.append((mpath.Path.MOVETO, p1))
        path_data.extend([(mpath.Path.CURVE3, cp), (mpath.Path.CURVE3, p2)])
        mx2, my2 = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        vx, vy = p2[0]-p1[0], p2[1]-p1[1]
        d = math.sqrt(vx**2+vy**2)
        if d > 0:
            nx_b, ny_b = -vy/d, vx/d
            if nx_b*(gx-mx2)+ny_b*(gy-my2) < 0: nx_b, ny_b = -nx_b, -ny_b
            ax.text(mx2-nx_b*grille*0.055, my2-ny_b*grille*0.055,
                    f"{courbures[bords[i]]}%", fontsize=7, color='#CC2233',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#FD3548', alpha=0.85), zorder=7)
    codes, verts = zip(*path_data)
    bezier_path = mpath.Path(verts, codes)
    ax.add_patch(mpatches.PathPatch(bezier_path, fc="#FD3548", alpha=0.35, ec="none", lw=0, zorder=3))
    patch_bleu = mpatches.PathPatch(bezier_path, fc="none", ec="#1a6fbf", lw=1.8, zorder=4)
    ax.add_patch(patch_bleu)
    
    # --- LOGIQUE TENTMESH : REPLI COURBÉ 7cm EN POINTILLÉ GRIS ---
    VAL_REPLI = 7.0  # Ta nouvelle valeur de 7 cm
    
    if tissu_select == "Tentmesh":
        for i in range(nb_pts):
            p1, p2 = p_fab[i], p_fab[(i+1)%nb_pts]
            
            # 1. On récupère la flèche de Bézier de la voile finie (ligne bleue)
            cp_fini, _ = fleche_bezier(p1, p2, courbures[bords[i]], gx, gy)
            
            # 2. Calcul du vecteur normal pour décaler vers l'extérieur
            vx, vy = p2[0]-p1[0], p2[1]-p1[1]
            d = math.sqrt(vx**2+vy**2)
            nx, ny = -vy/d, vx/d
            # On s'assure que le décalage va vers l'extérieur de la voile
            if nx*(gx-(p1[0]+p2[0])/2) + ny*(gy-(p1[1]+p2[1])/2) > 0:
                nx, ny = -nx, -ny
            
            # 3. Calcul des points de la ligne de COUPE (décalage de 7cm)
            # On ajoute le "biais" (prolongement) aux extrémités pour le repli des angles
            ext_p1 = (p1[0] + nx*VAL_REPLI*s - (vx/d)*VAL_REPLI*s, 
                      p1[1] + ny*VAL_REPLI*s - (vy/d)*VAL_REPLI*s)
            
            ext_p2 = (p2[0] + nx*VAL_REPLI*s + (vx/d)*VAL_REPLI*s, 
                      p2[1] + ny*VAL_REPLI*s + (vy/d)*VAL_REPLI*s)
            
            # Le point de contrôle de la courbe est aussi décalé de 7cm
            ext_cp = (cp_fini[0] + nx*VAL_REPLI*s, cp_fini[1] + ny*VAL_REPLI*s)
            
            # 4. Tracé de la courbe de coupe (Pointillé gris)
            courbe_coupe = mpath.Path([ext_p1, ext_cp, ext_p2], 
                                     [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])
            
            ax.add_patch(mpatches.PathPatch(courbe_coupe, fc="none", ec="#888888", 
                                            lw=1.2, ls='--', zorder=5, alpha=0.8))
            
            # 5. Tracé des biais aux angles (pour fermer le gabarit de découpe)
            ax.plot([p1[0], ext_p1[0]], [p1[1], ext_p1[1]], color='#888888', lw=1, ls='--', alpha=0.5)
            ax.plot([p2[0], ext_p2[0]], [p2[1], ext_p2[1]], color='#888888', lw=1, ls='--', alpha=0.5)    
    
    # Renforts — arcs concentriques par angle (une couche = un arc)
    R_MIN = grille * 0.04
    for i in range(nb_pts):
        pt = LABELS[i]
        if pt not in renforts_cm: continue
        rayons = renforts_cm[pt]
        px, py = p_fab[i]
        for r_cm in rayons:
            r = max(r_cm * s, R_MIN)
            circ = plt.Circle((px, py), r, color='#1a6fbf', fill=False,
                               lw=1.2, linestyle='-', alpha=0.7, zorder=8)
            ax.add_patch(circ)
            circ.set_clip_path(patch_bleu)
    # Laizes — orientation + calcul nb panneaux
    p1_ref = coords_fab[idx_cote_laize]
    p2_ref = coords_fab[(idx_cote_laize+1)%nb_pts]
    angle  = math.atan2(p2_ref[1]-p1_ref[1], p2_ref[0]-p1_ref[0])
    nx_l, ny_l = -math.sin(angle), math.cos(angle)
    dx_l, dy_l = math.cos(angle), math.sin(angle)
    pts_courbe = []
    for i in range(nb_pts):
        p1, p2 = p_fab[i], p_fab[(i+1)%nb_pts]
        cp, _ = fleche_bezier(p1, p2, courbures[bords[i]], gx, gy)
        for t in np.linspace(0, 1, 30):
            pts_courbe.append(((1-t)**2*p1[0]+2*(1-t)*t*cp[0]+t**2*p2[0],
                               (1-t)**2*p1[1]+2*(1-t)*t*cp[1]+t**2*p2[1]))
    projs = [p[0]*nx_l + p[1]*ny_l for p in pts_courbe]
    d_min, d_max = min(projs), max(projs)
    largeur_totale = d_max - d_min
    if largeur_bdf_total is None:
        largeur_bdf_total = nb_bandes_rouleau * largeur_bdf
    largeur_utile = laize_theorique_cm - 2.0 - largeur_bdf_total
    pas_max_physique = (largeur_utile - recouvr_actuel) * s

    # --- LOGIQUE DE DÉCOUPE DES LAIZES AVEC ÉQUILIBRAGE ---
    largeur_max_dispo_cm = laize_theorique_cm - 2.0 - largeur_bdf_total
    pas_max_physique = (largeur_max_dispo_cm - recouvr_actuel) * s

    list_pas = []
    cumul = 0
    
    if mode_economique:
    # --- MODE ÉCONOMIQUE : reste en premier (vers le sommet/renfort), puis laizes pleines ---
        nb_full = int(largeur_totale / pas_max_physique)
        reste_initial = largeur_totale - nb_full * pas_max_physique
    
        if reste_initial > 1e-7:
            seuil_mini = 0.40 * pas_max_physique
    
            # Hauteur du premier panneau (le plus petit, au sommet)
            d_end_reste = d_min + reste_initial
            pts_reste = [p for p in pts_courbe
                         if d_min - 1e-7 <= p[0]*nx_l + p[1]*ny_l <= d_end_reste + 1e-7]
            hauteur_reste = 0
            if pts_reste:
                longs_r = [p[0]*dx_l + p[1]*dy_l for p in pts_reste]
                hauteur_reste = (max(longs_r) - min(longs_r)) / s
    
            if reste_initial < seuil_mini and nb_full >= 1 and hauteur_reste > 100:
                # Équilibrage : les deux premiers panneaux équilibrés
                largeur_equilibree = (pas_max_physique + reste_initial) / 2
                list_pas = [largeur_equilibree, largeur_equilibree] + [pas_max_physique] * (nb_full - 1)
            else:
                # Petit panneau en premier → couvert par le renfort du sommet
                list_pas = [reste_initial] + [pas_max_physique] * nb_full
        else:
            list_pas = [pas_max_physique] * max(nb_full, 1)
    else:
        # --- MODE ESTHÉTIQUE : Parts égales sans dépasser la laize max ---
        # 1. On calcule combien de laizes entières max il faut au minimum
        nb_laizes_necessaires = math.ceil(largeur_totale / pas_max_physique)
        
        # Sécurité : au moins 1 laize
        nb_laizes_necessaires = max(1, nb_laizes_necessaires) 
        
        # 2. On divise la largeur totale en parts strictement égales
        pas_identique = largeur_totale / nb_laizes_necessaires
        list_pas = [pas_identique] * nb_laizes_necessaires

    # Calcul final des largeurs de coupe (cm)
    list_W = [round((p/s) + recouvr_actuel, 1) for p in list_pas]
    nb_laizes = len(list_pas)
    

    # Longueurs réelles par panneau (pour le bilan matière)
    list_L = []
    cumul  = 0
    for k, pas in enumerate(list_pas):
        d_start = d_min + cumul
        d_end   = d_start + pas
        d_mid   = (d_start + d_end) / 2
        cumul  += pas
        if k < nb_laizes - 1:
            cx_l, cy_l = d_end*nx_l, d_end*ny_l
            line, = ax.plot([cx_l-dx_l*grille, cx_l+dx_l*grille],
                            [cy_l-dy_l*grille, cy_l+dy_l*grille],
                            color='#1a6fbf', lw=1.1, alpha=0.6, zorder=5)
            line.set_clip_path(patch_bleu)
        pts_laize = [p for p in pts_courbe if d_start <= p[0]*nx_l+p[1]*ny_l <= d_end]
        if pts_laize:
            longs = [p[0]*dx_l+p[1]*dy_l for p in pts_laize]
            l_pos = (min(longs)+max(longs)) / 2
            fx = d_mid*nx_l + l_pos*dx_l
            fy = d_mid*ny_l + l_pos*dy_l
            ax.text(fx, fy, str(k+1), color='#1a6fbf', fontsize=9,
                    fontweight='bold', ha='center', va='center', zorder=15)
            list_L.append((max(longs)-min(longs)) / s)
        else:
            list_L.append(0.0)
            
    
    return ax, nb_laizes, list_W, list_L, list_pas

# =============================================================================
# 6. BARRE LATÉRALE
# =============================================================================
with st.sidebar:
    try:
        
        st.image("/Users/jadeleroux/Documents/Polytech/Stage 4A/Voilerie Tarot/Logo/Voilerie-Tarot-Logo-Horizontal.png", width=200)
    except Exception:
        st.markdown("### Voilerie Tarot")
    st.header("⚙️ Paramètres")
    nom_client      = st.text_input("👤 Client", placeholder="Nom du dossier")
    date_aujourdhui = date.today().strftime("%d/%m/%Y")
    st.markdown(f"**Date :** {date_aujourdhui}")
    c_lbl, c_pick = st.columns([1.3, 1.5])
    c_lbl.markdown(
        "<div style='display:flex; align-items:center; height:38px;'><b>Référence :</b></div>",
        unsafe_allow_html=True)
    reference = c_pick.text_input(
        "Référence",
        placeholder="Ex : VO1",
        key="reference",
        label_visibility="collapsed")
    st.divider()
    
    # Tissu et laize
    tissu_select = st.selectbox("**Tissu**", list(MATERIAUX.keys()))
    
    # Choix automatique du recouvrement selon le tissu
    recouvr_actuel = RECOUVR_TENT if tissu_select == "Tentmesh" else RECOUVR_STD
    if tissu_select == "Autre...":
        tissu           = st.text_input("Nom du matériau", placeholder="ex: Sunbrella", key="nom_tissu_perso")
        laize_choisie   = st.number_input("Largeur laize brute (cm)", value=150, step=1, key="laize_perso")
        poids_surfacique = st.number_input("Poids surfacique (g/m²)", value=300, step=10, key="poids_perso")
        long_rouleau_m  = st.number_input("Longueur du rouleau (m)", value=50, step=5, key="long_rouleau_perso")
        long_rouleau_cm = long_rouleau_m * 100
    else:
        tissu         = tissu_select
        c1, c2 = st.columns([5, 1])
        with c1:
            laize_choisie = st.selectbox("**Laize**", MATERIAUX[tissu]["laizes"], key="laize_standard")
        with c2:
            st.markdown("<div style='margin-top:42px;color:#888;font-size:0.9em;'>cm</div>", unsafe_allow_html=True)
    
    # --- GESTION DYNAMIQUE DE LA DIVISION ---
    division = 1
    tissus_divisibles = ["Tentmesh", "Meaban"]

    if tissu_select in tissus_divisibles:
        # On affiche la case à cocher uniquement pour ces deux tissus
        if st.checkbox(f"✂Diviser la laize de {tissu_select} en deux", value=False, key="opt_division"):
            division = 2
    
    # Calcul de la laize utile (toujours valide car division vaut au moins 1)
    laize_utile_cm = laize_choisie / division
    
    # Affichage de l'information de découpe si nécessaire
    if division > 1:
        st.info(f"Mode 1/2 laize actif : **{laize_utile_cm:.1f} cm**")
    
    
    # 2. LECTURE DYNAMIQUE PAR COLONNE DEPUIS EXCEL
    try:
        from pathlib import Path
        chemin_excel = Path(__file__).parent.parent / "referentiel_materiaux.xlsx"
        
        # On charge le fichier Excel (la ligne 1 devient automatiquement les noms des colonnes)
        df_coloris = pd.read_excel(chemin_excel, sheet_name="Coloris")
        
        if tissu_select == "Autre...":
            coloris = st.text_input("**Coloris du tissu**", placeholder="Ex: Turquoise", key="coloris_final")
        else:
            # On vérifie si le nom du tissu sélectionné (ex: "Soltis 92") correspond bien à une colonne Excel
            if tissu_select in df_coloris.columns:
                # On récupère la colonne, on nettoie les cases vides du bas (dropna) et on liste les couleurs
                couleurs_dispo = df_coloris[tissu_select].dropna().unique().tolist()
                
                if couleurs_dispo:
                    coloris = st.selectbox("**Coloris du tissu**", options=couleurs_dispo, key="coloris_final")
                else:
                    coloris = st.text_input("**Coloris (Colonne vide dans Excel)**", placeholder="Écrire la couleur", key="coloris_final")
            else:
                # Sécurité si l'orthographe dans MATERIAUX et l'en-tête Excel ne match pas tout à fait
                coloris = st.text_input("**Coloris**", placeholder="Écrire la couleur", key="coloris_final")
                
    except Exception as e:
        st.error(f"Erreur de liaison Excel : {e}")
        coloris = st.text_input("Coloris du tissu", placeholder="Blanc Neige", key="coloris_final")
    
    # --- BOUTON TÉLÉCHARGEMENT DU NUANCIER DANS LA BARRE LATÉRALE ---
    if tissu_select in NUANCIERS_PDF:
        nom_fichier = NUANCIERS_PDF[tissu_select]
        chemin_complet = os.path.join(DOSSIER_NUANCIERS, nom_fichier)
    
        if os.path.exists(chemin_complet):
            with open(chemin_complet, "rb") as f:
                st.download_button(
                    label=f"📑 Nuancier {tissu_select}",
                    data=f.read(),
                    file_name=nom_fichier,
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_nuancier_{tissu_select}",
                )
        else:
            st.warning(f"Nuancier introuvable : {chemin_complet}")
            
    style_label = "<div style='margin-top:10px; font-weight:bold;'>"
    style_cm = "<div style='margin-top:10px; color:#888; font-size:0.9em;'>"
    
    nb_points = st.radio("**Nombre de côtés**", [3, 4, 5], index=1, horizontal=True)

    if tissu_select != "Tentmesh":
        st.divider()
        st.subheader("Bandes droit Fil")
        nb_bandes_rouleau = st.radio("**-  Nombre :**", [1, 2, 3, 4], horizontal=True)
        
        st.markdown("##### **-  Largeur (cm) :**")
        largeurs_bdf = []
        cols_bdf = st.columns(nb_bandes_rouleau)
        for k in range(nb_bandes_rouleau):
            w = cols_bdf[k].selectbox(f"BDF #{k+1}",[7.0, 8.0, 10.0],index=0,key=f"bdf_w_{k}",)
            largeurs_bdf.append(w)
        largeur_bdf_total = sum(largeurs_bdf)
        largeur_bdf_choisie = largeur_bdf_total / nb_bandes_rouleau
        
        st.markdown("##### **-  Longueur (m) :**")
        longueurs_bdf = []
        cols_bdf_l = st.columns(nb_bandes_rouleau)
        for k in range(nb_bandes_rouleau):
            l = cols_bdf_l[k].number_input(
                f"BDF L#{k+1}", min_value=0.0, value=0.0, step=0.5,
                key=f"bdf_l_{k}", label_visibility="collapsed",
                help="0 = calcul auto")
            longueurs_bdf.append(l)
    else:
        nb_bandes_rouleau = 0
        largeur_bdf_total = 0
        largeur_bdf_choisie = 0
        longueur_bdf_manuelle_m = 0.0
        longueurs_bdf = []
        largeurs_bdf = []
    
    # 2. STRATÉGIE ==========================================
    st.divider()
    c1, c2 = st.columns([2.5, 4])
    with c1:
        # On remplace le subheader par un label classique pour l'alignement
        st.markdown(f"{style_label}Stratégie :</div>", unsafe_allow_html=True)
    with c2:
        strategie = st.radio(
            "Priorité", 
            ["Esthétique", "Économique"], 
            index=0, 
            label_visibility="collapsed",
            horizontal=True
        )
    mode_economique = (strategie == "Économique")
    bords = BORDS_PAR_NB[nb_points]

    # 3. LONGUEURS MUR À MUR==========================================
    st.divider()
    st.subheader("Longueurs Mur à Mur")
    mesures = {}
    
    # Fonction d'input ajustée à [1.5, 4, 1] pour matcher le haut
    def input_mesure(label, key, defaut=500):
        c1, c2, c3 = st.columns([1.5, 4, 1])
        c1.markdown(f"{style_label}{label} :</div>", unsafe_allow_html=True)
        valeur = c2.number_input(label, value=defaut, key=key, label_visibility="collapsed")
        c3.markdown(f"{style_cm}cm</div>", unsafe_allow_html=True)
        return valeur
    
    for b in bords: 
        mesures[b] = input_mesure(b, f"m_{b}")
        
    if nb_points == 4: 
        mesures["BD"] = input_mesure("BD", "m_bd", 667)
    if nb_points == 5:
        mesures["AC"] = input_mesure("AC", "m_ac", 700)
        mesures["AD"] = input_mesure("AD", "m_ad", 750)
    
    # Diagonale de vérification alignée
    diag_info = None
    if nb_points == 4:
        cp = generer_coords_quadrilatere(mesures["AB"], mesures["BC"], mesures["CD"], mesures["DA"], mesures["BD"])
        if cp: diag_info = ("AC", longueur_cote(cp[0], cp[2]))
    elif nb_points == 5:
        cp = generer_coords_pentagone(mesures["AB"], mesures["BC"], mesures["CD"], mesures["DE"], mesures["EA"], mesures["AC"], mesures["AD"])
        if cp: diag_info = ("BD", longueur_cote(cp[1], cp[3]))
    
    if diag_info:
        label, val = diag_info
        c_lbl, c_val, c_unit = st.columns([1.5, 4, 1])
        c_lbl.markdown(f"{style_label}{label} :</div>", unsafe_allow_html=True)
        c_val.markdown(
            f"<div style='margin-top:6px; padding:8px 12px; background:rgba(255,255,255,0.04); border-radius:6px; color:#999;'>"
            f"{val:.1f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        c_unit.markdown(f"{style_cm}cm</div>", unsafe_allow_html=True)
    
    # 4. ACCASTILLAGE PAR ANGLE
    st.divider()
    st.subheader("Accastillage par Angle")
    surf_estimee = st.session_state.get('_surf_m2_prec', 0.0)
    if 0 < surf_estimee < SEUIL_PETITE_VOILE_M2:
        st.caption("⚠️ Voile < 10 m² : privilégier des palans simples.")
    config_angles = {}
    fixation_angles = {}
    for pt in LABELS[:nb_points]:
        # Changement ici pour respecter les proportions [1.5, 4, 1.5]
        c_lbl, c_acc, c_val = st.columns([1.5, 4, 1.5])
        c_lbl.markdown(f"<div translate='no' style='margin-top:10px; font-weight:bold;'>{pt} :</div>", unsafe_allow_html=True)
        
        choix = c_acc.selectbox(
            f"Accastillage {pt}",
            list(ACCASTILLAGE.keys()),
            key=f"acc_{pt}",
            label_visibility="collapsed"
        )
        defaut = ACCASTILLAGE[choix]
        if st.session_state.get(f"acc_prev_{pt}") != choix:
            st.session_state[f"rf_{pt}"] = str(defaut)
            st.session_state[f"acc_prev_{pt}"] = choix
        
        retrait_str = c_val.text_input(
            "cm", 
            value=str(defaut), 
            key=f"rf_{pt}", 
            label_visibility="collapsed"
        )
        try:
            config_angles[pt] = float(retrait_str.replace(',', '.'))
        except ValueError:
            config_angles[pt] = float(defaut)
        c_lbl_f, c_fix = st.columns([1.5, 6])
        c_lbl_f.markdown(
            "<div style='margin-top:6px; color:#888; font-size:0.85em;'>Fixation :</div>",
            unsafe_allow_html=True)
        fixation_angles[pt] = c_fix.radio(
        f"Fixation {pt}",
        ["Mât", "Piton"],
        index=None,
        horizontal=True,
        key=f"fix_{pt}",
        label_visibility="collapsed"
    )
    # --- Configuration Emmagasineur ---
    indices_emm = [i for i, pt in enumerate(LABELS[:nb_points]) 
                  if st.session_state.get(f"acc_{pt}") in ACCASTILLAGE_EMM]
    
    is_emm = len(indices_emm) >= 2
    type_emm = "Aucun"
    emplacement_emm = ""

    if is_emm:
        st.divider()
        st.markdown("### ⚙️ Configuration Emmagasineur")
        
        i1, i2 = indices_emm[0], indices_emm[1]
        p1_label, p2_label = LABELS[i1], LABELS[i2]
        
        diff = abs(i1 - i2)
        est_bord = (diff == 1) or (diff == nb_points - 1)
        
        if est_bord:
            type_emm = "Sur Bord"
            emplacement_emm = f"{p1_label}{p2_label}" if diff == 1 else f"{p2_label}{p1_label}"
            if emplacement_emm not in bords:
                emplacement_emm = f"{p2_label}{p1_label}"
            
            # Affichage sobre en mode texte
            st.markdown(f"📍 Axe : **Bord {p1_label}-{p2_label}**")
            st.markdown(f"-> BDF 10 cm — laizes // à {emplacement_emm}")
        else:
            type_emm = "Diagonale"
            emplacement_emm = f"{p1_label}{p2_label}"
            
            st.markdown(f"📍 Axe : **Diagonale {p1_label}-{p2_label}**")
            st.markdown(f"-> BDF 8 cm sur l'axe {emplacement_emm}")

    # Orientation des laizes
    st.divider()
    st.subheader("Orientation des laizes")
    choix_orientation = st.radio("Référence", ["Automatique", "Manuel"],
                                  horizontal=True, key="orientation_laize", label_visibility="collapsed")
    if choix_orientation == "Manuel":
        cote_ref_manuel = st.selectbox(
            "Côté de référence",
            options=bords,
            format_func=lambda x: f"Bord {x}",
            key="cote_ref_laize"
        )
    else:
        cote_ref_manuel = None
    
    # --- CONFIGURATION DES COURBURES (CORRIGÉE) ---
    st.divider()
    st.subheader("Gestion des flèches")
    # 1. On calcule d'abord les points emmagasineur pour que 'courbures' y ait accès
    indices_emm = [i for i, pt in enumerate(LABELS[:nb_points]) 
                   if st.session_state.get(f"acc_{pt}") in ACCASTILLAGE_EMM]
    
    pts_emm = set(indices_emm)
    if len(indices_emm) >= 2 and type_emm == "Diagonale" and emplacement_emm in IDX_DIAG:
        pts_emm.update(IDX_DIAG[emplacement_emm])

    # 2. On affiche le sélecteur de mode
    choix_courbures = st.radio("Référence", ["Automatique", "Manuel"],
                                  horizontal=True, key="mode_courbure", label_visibility="collapsed")
    courbures = {}
    bords = BORDS_PAR_NB[nb_points]
    
    if choix_courbures == "Manuel":
        st.markdown("##### **- Pourcentage de flèche :**")
        cols_courbures = st.columns(min(len(bords), 3))
        for idx, b in enumerate(bords):
            col_idx = idx % 3
            # Valeur par défaut : 1.5% si emmagasineur sur ce bord, sinon 3.0%
            val_defaut = 1.5 if (len(indices_emm) >= 2 and type_emm == "Sur Bord" and b == emplacement_emm) else 3.0
            
            with cols_courbures[col_idx]:
                courbures[b] = st.slider(
                    f"Bord {b}", 
                    min_value=0.0, 
                    max_value=10.0, 
                    value=val_defaut, 
                    step=0.5,
                    key=f"slide_courbures_{b}"
                )
    else:
        # Mode automatique : utilise les points calculés juste au-dessus
        for i, b in enumerate(bords):
            courbures[b] = 3.0
            if len(indices_emm) >= 2 and emplacement_emm in courbures: 
                courbures[emplacement_emm] = 1.5
            if i in pts_emm and (i+1)%nb_points in pts_emm: 
                courbures[b] = 1.5
# =============================================================================
# 7. CALCULS GÉOMÉTRIQUES
# =============================================================================
coords_mur = {
    3: lambda: generer_coords_triangle(mesures["CA"], mesures["BC"], mesures["AB"]),
    4: lambda: generer_coords_quadrilatere(mesures["AB"], mesures["BC"], mesures["CD"], mesures["DA"], mesures["BD"]),
    5: lambda: generer_coords_pentagone(mesures["AB"], mesures["BC"], mesures["CD"], mesures["DE"], mesures["EA"], mesures["AC"], mesures["AD"]),
}[nb_points]()
surf_m2 = aire_pleine = 0.0
coords_fab = coords_visu_rouge = None
longueurs_arcs_metres = {}
cotes_fabrication = {}
nb_laizes = 0; renforts_cm = {}
labels = list(LABELS[:nb_points])
# Points emmagasineur
pts_emm = set(indices_emm)
if is_emm and type_emm == "Diagonale" and emplacement_emm in IDX_DIAG:
    pts_emm.update(IDX_DIAG[emplacement_emm])
if coords_mur:
    gx = sum(p[0] for p in coords_mur) / nb_points
    gy = sum(p[1] for p in coords_mur) / nb_points
    # Axe emmagasineur
    idx_emm_axis = []
    if is_emm and emplacement_emm:
        if type_emm == "Sur Bord":
            idx_emm_axis = [indices_emm[0], indices_emm[1]]
        else:
            idx_emm_axis = list(IDX_DIAG.get(emplacement_emm, []))
    # Points de fabrication
    coords_fab = [None] * nb_points
    coords_visu_rouge = [None] * nb_points
    if len(idx_emm_axis) == 2:
        i1, i2 = idx_emm_axis
        P1, P2 = coords_mur[i1], coords_mur[i2]
        d = longueur_cote(P1, P2)
        if d > 0:
            ux, uy = (P2[0]-P1[0])/d, (P2[1]-P1[1])/d
            coords_fab[i1] = (P1[0]+ux*config_angles[labels[i1]], P1[1]+uy*config_angles[labels[i1]])
            coords_fab[i2] = (P2[0]-ux*config_angles[labels[i2]], P2[1]-uy*config_angles[labels[i2]])
            coords_visu_rouge[i1], coords_visu_rouge[i2] = P1, P2
    for i in range(nb_points):
        if coords_fab[i] is None:
            P = coords_mur[i]; R = config_angles[labels[i]]
            vx, vy = gx-P[0], gy-P[1]; d = math.sqrt(vx**2+vy**2)
            coords_fab[i] = (P[0]+(vx/d)*R, P[1]+(vy/d)*R) if d > 0 else P
            coords_visu_rouge[i] = coords_fab[i]
    # Courbures
    # courbures = {b: 3.0 for b in bords}
    # if is_emm and emplacement_emm in courbures: courbures[emplacement_emm] = 1.5
    # for i, b in enumerate(bords):
    #     if i in pts_emm and (i+1)%nb_points in pts_emm: courbures[b] = 1.5
    
    
    # Longueurs et surfaces
    for i in range(nb_points):
        b = bords[i]
        cotes_fabrication[b] = longueur_cote(coords_fab[i], coords_fab[(i+1)%nb_points])
        longueurs_arcs_metres[b] = calculer_longueur_arc(cotes_fabrication[b], courbures[b]) / 100
    if nb_points == 4:
        cotes_fabrication["BD"] = longueur_cote(coords_fab[1], coords_fab[3])
    elif nb_points == 5:
        cotes_fabrication["AC"] = longueur_cote(coords_fab[0], coords_fab[2])
        cotes_fabrication["AD"] = longueur_cote(coords_fab[0], coords_fab[3])
    cf = cotes_fabrication
    if nb_points == 3:
        aire_pleine = aire_triangle(cf["AB"], cf["BC"], cf["CA"])
    elif nb_points == 4:
        aire_pleine = aire_triangle(cf["AB"], cf["DA"], cf["BD"]) + aire_triangle(cf["BC"], cf["CD"], cf["BD"])
    elif nb_points == 5:
        aire_pleine = (aire_triangle(cf["AB"], cf["BC"], cf["AC"]) +
                       aire_triangle(cf["AC"], cf["CD"], cf["AD"]) +
                       aire_triangle(cf["AD"], cf["DE"], cf["EA"]))
    aire_perdue = sum((2/3)*cf[b]*(courbures[b]/100)*cf[b] for b in bords)
    surf_m2 = (aire_pleine - aire_perdue) / 10000
    st.session_state['_surf_m2_prec'] = surf_m2
    if nb_points == 4:
        longueurs_arcs_metres["AC"] = longueur_cote(coords_fab[0], coords_fab[2]) / 100
        longueurs_arcs_metres["BD"] = longueur_cote(coords_fab[1], coords_fab[3]) / 100
    elif nb_points == 5:
        longueurs_arcs_metres["AC"] = longueur_cote(coords_fab[0], coords_fab[2]) / 100
        longueurs_arcs_metres["AD"] = longueur_cote(coords_fab[0], coords_fab[3]) / 100
        longueurs_arcs_metres["BD"] = longueur_cote(coords_fab[1], coords_fab[3]) / 100
        longueurs_arcs_metres["CE"] = longueur_cote(coords_fab[2], coords_fab[4]) / 100
    N_SEG = 256
    pts_full_app = []
    for _i in range(nb_points):
        _p1 = coords_fab[_i]
        _p2 = coords_fab[(_i+1) % nb_points]
        _cp = fleche_bezier(_p1, _p2, courbures[bords[_i]], gx, gy)[0]
        for _k in range(N_SEG):
            _t = _k / N_SEG
            pts_full_app.append((
                (1-_t)**2*_p1[0] + 2*(1-_t)*_t*_cp[0] + _t**2*_p2[0],
                (1-_t)**2*_p1[1] + 2*(1-_t)*_t*_cp[1] + _t**2*_p2[1]
            ))
    n_total_app = len(pts_full_app)
    renforts_cm = calculer_renforts(coords_fab, nb_points, pts_emm, surf_m2, pts_full_app, N_SEG)
    renforts_cm_fab = preparer_renforts_fabrication(renforts_cm)
    
    # ## --- VÉRIFICATION DU SUR-ENROULEMENT ---
    if nb_points == 3:
        alerte_enroulement = False
        details_alerte = []
        
        for i, pt in enumerate(LABELS[:nb_points]):
            choix_acc = st.session_state.get(f"acc_{pt}", "")
            
            # Détection robuste du matériel d'enroulement
            if any(nom in choix_acc for nom in ["VOGE", "VOPE"]):
                p_prev = coords_fab[(i-1) % nb_points]
                p_curr = coords_fab[i]
                p_next = coords_fab[(i+1) % nb_points]
                # Angle Bézier (voile finie)
                gx_disp = sum(p[0] for p in coords_fab) / nb_points
                gy_disp = sum(p[1] for p in coords_fab) / nb_points
                cp_arr_d = fleche_bezier(p_prev, p_curr, courbures.get(bords[(i-1)%nb_points], 0.0), gx_disp, gy_disp)[0]
                cp_dep_d = fleche_bezier(p_curr, p_next, courbures.get(bords[i],               0.0), gx_disp, gy_disp)[0]
                def _ud(dx, dy):
                    d = math.sqrt(dx**2+dy**2)
                    return (dx/d, dy/d) if d > 1e-9 else (1.0, 0.0)
                tv1d = _ud(cp_arr_d[0]-p_curr[0], cp_arr_d[1]-p_curr[1])
                tv2d = _ud(cp_dep_d[0]-p_curr[0], cp_dep_d[1]-p_curr[1])
                ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, tv1d[0]*tv2d[0]+tv1d[1]*tv2d[1]))))
                
                # Seuil critique supérieur à 88°
                if ang_deg > 88.0:
                    alerte_enroulement = True
                    nom_composant = "VOPE" if "VOPE" in choix_acc else "VOGE"
                    # On combine l'angle et le conseil dans la même chaîne de caractères
                    details_alerte.append(f"Angle au sommet {pt}' = {ang_deg:.1f}° — Inversez VOGE et VOPE")

        # --- AFFICHAGE DE L'ALERTE DANS L'INTERFACE ---
        if alerte_enroulement:
            # Titre général en rouge critique
            st.markdown(
                "<h4 style='color: #FF0000; margin-bottom: 5px;'>⚠️ Risque de sur-enroulement critique !</h4>", 
                unsafe_allow_html=True
            )
            # Affichage des détails textuels des angles fautifs
            for detail in details_alerte:
                st.error(detail)
# =============================================================================
# 8. BANDES DROIT FIL
# =============================================================================
perimetre_arcs_cm = sum(longueurs_arcs_metres.values()) * 100
if is_emm and type_emm == "Sur Bord":
    largeur_bdf = 10.0
    longueur_bdf_cm = perimetre_arcs_cm * COEFF_BDF
elif is_emm and type_emm == "Diagonale":
    largeur_bdf = 8.0
    i1, i2 = IDX_DIAG.get(emplacement_emm, (0,2))
    longueur_bdf_cm = (longueur_cote(coords_fab[i1], coords_fab[i2]) if coords_fab
                       else mesures.get(emplacement_emm, 0)) * COEFF_BDF
else:
    largeur_bdf = 7.0
    longueur_bdf_cm = perimetre_arcs_cm * COEFF_BDF

# Si au moins une longueur manuelle est renseignée, on prend la somme des non-nulles
longueurs_bdf_saisies = [l for l in longueurs_bdf if l > 0]
if longueurs_bdf_saisies:
    longueur_bdf_cm = sum(longueurs_bdf_saisies) * 100
# =============================================================================
# 9. LAIZES
# =============================================================================

# On utilise 'recouvr_actuel' pour la laize nette
laize_nette_cm = laize_choisie - 2.0 - nb_bandes_rouleau * largeur_bdf - recouvr_actuel
if choix_orientation == "Manuel" and cote_ref_manuel in bords:
    idx_cote_laize = bords.index(cote_ref_manuel)
elif is_emm and type_emm == "Sur Bord" and emplacement_emm in bords:
    idx_cote_laize = bords.index(emplacement_emm)
else:
    idx_cote_laize = max(range(len(bords)), key=lambda i: cotes_fabrication.get(bords[i], 0))
# =============================================================================
# 10. CALCULS D'AFFICHAGE (avant les colonnes pour le PDF)
# =============================================================================
list_W        = []
list_L        = []
nb_laizes     = 0
largeur_dispo = laize_choisie - largeur_bdf_total - 2.0
if tissu_select == "Tentmesh":
    formule_bdf = "Aucune (Bord de sécurité 2cm uniquement)"
else:
    from collections import Counter
    compteur = Counter(largeurs_bdf)
    formule_bdf = " + ".join(
        f"{n} bande{'s' if n > 1 else ''} de {w:g}cm"
        for w, n in sorted(compteur.items())
    )
laize_a_couper = 0.0
reste_final    = largeur_dispo
# =============================================================================
# 11. CALCULS DES SURFACES ADDITIONNELLES
# =============================================================================
# Construction des segments (points de contrôle Bézier par côté)
segments = []
for i in range(nb_points):
    p1 = coords_fab[i]
    p2 = coords_fab[(i+1) % nb_points]
    cp, _ = fleche_bezier(p1, p2, courbures[bords[i]], gx, gy)
    segments.append({'p1': p1, 'cp': cp, 'p2': p2})
# Surface des renforts
surf_renforts_m2 = 0.0
if coords_fab:
    for i, pt in enumerate(LABELS[:nb_points]):
        if pt not in renforts_cm_fab:
            continue
        p_curr = coords_fab[i]
        seg_in  = segments[(i-1) % nb_points]
        seg_out = segments[i]
        for r in renforts_cm_fab[pt]:
            surf_renforts_m2 += calculer_aire_renfort(
                p_curr, seg_in['cp'], seg_out['cp'],
                seg_in['p1'], seg_out['p2'], r
            )
    surf_renforts_m2 /= 10000

# Surface BDF (bande par bande)
surf_bdf_m2 = 0.0
if coords_fab:
    if tissu_select == "Tentmesh":
        # RÈGLE TENTMESH : On calcule le surplus de 7cm sur le périmètre fini
        perimetre_fini_cm = sum(longueurs_arcs_metres.values()) * 100
        surf_bdf_m2 = (perimetre_fini_cm * 7.0) / 10000
    else:
        # RÈGLE STANDARD : Bandes Droit Fil classiques
        temp_surf_cm2 = 0.0
        for b in bords:
            long_arc_cm = longueurs_arcs_metres[b] * 100
            # Largeur de 10cm si emmagasineur sur bord, sinon 7cm
            largeur = 10.0 if (is_emm and type_emm == "Sur Bord" and b == emplacement_emm) else 7.0
            temp_surf_cm2 += (long_arc_cm * largeur)
        
        # Ajout de la diagonale si emmagasineur central
        if is_emm and type_emm == "Diagonale":
            temp_surf_cm2 += (longueur_bdf_cm * 8.0)
            
        surf_bdf_m2 = temp_surf_cm2 / 10000
        
    surf_bdf_saisie_m2 = 0.0
    for k in range(nb_bandes_rouleau):
        if longueurs_bdf[k] > 0:
            surf_bdf_saisie_m2 += longueurs_bdf[k] * (largeurs_bdf[k] / 100)
    


    surf_materiau_m2 = surf_m2 + surf_renforts_m2 + surf_bdf_saisie_m2

else:
    # Si la géométrie de la voile plante, on met tout à 0
    surf_renforts_m2 = surf_bdf_m2 = debit_total_atelier = 0.0
    surf_materiau_m2 = 0.0

# =============================================================================
# 12. AFFICHAGE
# =============================================================================
st.markdown("""
    <style>
    /* Taille des metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.0rem !important;
    }
    /* Réduire l'espace interne (padding) du container de la fiche technique */
    [data-testid="stVerticalBlockBorderContainer"] {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    /* Resserrer l'espace au-dessus et en-dessous des dividers (lignes horizontales) */
    hr {
        margin-top: 0.8rem !important;
        margin-bottom: 0.8rem !important;
    }
    /* Réduire l'espace entre les différents blocs verticaux de Streamlit */
    [data-testid="stVerticalBlock"] {
        gap: 0.6rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
if coords_fab:
    poids_surfacique = st.session_state.get("poids_perso", 300) if tissu_select == "Autre..." else MATERIAUX[tissu_select]["poids"]
    masse_totale_kg = (surf_materiau_m2 * poids_surfacique) / 1000
else:
    poids_surfacique = 0
    masse_totale_kg = 0.0
    
col_visu, col_recap = st.columns([1.8, 1.2], gap="medium")
pertes_pct = 0.0
pertes_m2  = 0.0
with col_visu: 
    st.subheader("Visualisation")

    if coords_mur and coords_fab:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax, nb_laizes, list_W, list_L, list_pas = tracer_voile(
            ax, coords_fab, coords_visu_rouge, coords_mur,
            courbures, bords, nb_points, laize_nette_cm,
            idx_cote_laize, renforts_cm, pts_emm,
            laize_theorique_cm=laize_utile_cm,
            mode_economique=mode_economique,
            nb_bandes_rouleau=nb_bandes_rouleau,
            largeur_bdf=largeur_bdf_choisie,
            largeur_bdf_total=largeur_bdf_total)
        fig.tight_layout(); st.pyplot(fig)
        # if list_L:
        #     nb_pan = len(list_L)
        #     long_rouleau_cm = sum(list_L) + nb_pan * 1.0
        #     surf_tissu_reelle = (long_rouleau_cm * laize_choisie) / 10000
        #     surf_utile = surf_m2 + surf_renforts_m2 + surf_bdf_m2
        #     pertes_m2  = max(0.0, surf_tissu_reelle - surf_utile)
        #     pertes_pct = (pertes_m2 / surf_tissu_reelle * 100) if surf_tissu_reelle > 0 else 0.0
        # laize_a_couper = max(list_W) if list_W else 0
        # reste_final    = largeur_dispo - laize_a_couper
        from generate_dxf_tarot import (extract_laize_contour, full_contour_polyline,
                                  build_sail_segments, laize_axes as laize_axes_dxf)

        def shoelace_cm2(contour):
            n = len(contour)
            a = sum(contour[i][0]*contour[(i+1)%n][1] - contour[(i+1)%n][0]*contour[i][1]
                    for i in range(n))
            return abs(a) / 2
        
        if list_W and list_L and coords_fab:
            segs     = build_sail_segments(coords_fab, courbures, bords, nb_points)
            pts_full = full_contour_polyline(segs, n_per_seg=256)
            nx_l_d, ny_l_d, dx_l_d, dy_l_d = laize_axes_dxf(coords_fab, idx_cote_laize, nb_points)
            projs    = [p[0]*nx_l_d + p[1]*ny_l_d for p in pts_full]
            d_min_d, d_max_d = min(projs), max(projs)
        
            sum_pas      = sum(list_pas)
            list_pas_cm  = [float(p)*(d_max_d - d_min_d)/sum_pas for p in list_pas]
        
            surf_laizes_m2 = 0.0
            cumul = 0.0
            for k, pas_v in enumerate(list_pas_cm):
                d_start    = d_min_d + cumul
                d_end      = d_start + pas_v + recouvr_actuel if k < len(list_pas_cm)-1 else d_max_d
                contour_cm, _ = extract_laize_contour(pts_full, nx_l_d, ny_l_d, dx_l_d, dy_l_d, d_start, d_end)
                if contour_cm:
                    surf_laizes_m2 += shoelace_cm2(contour_cm) / 10000
                cumul += pas_v
        
            surf_tissu_reelle = surf_laizes_m2
            surf_materiau_m2  = surf_laizes_m2 + surf_renforts_m2 + surf_bdf_saisie_m2
            pertes_m2  = max(0.0, surf_tissu_reelle - (surf_m2 + surf_renforts_m2 + surf_bdf_m2))
            pertes_pct = (pertes_m2 / surf_tissu_reelle * 100) if surf_tissu_reelle > 0 else 0.0
    else:
        st.error("⚠️ Géométrie impossible — vérifiez les dimensions.")
type_voile = {3: "Triangulaire", 4: "Quadrangulaire", 5: "Pentagonale"}.get(nb_points, "")
with col_recap:
    c1, c2 = st.columns(2)
    if coords_fab is not None:
        try:
            from generate_pdf_tarot import generer_pdf as _gen_pdf
            from generate_pdf_tarot import calculer_L_max_draille
            from generate_dxf_tarot import generer_zip_dxf

            # L_max pour les palans d'emmagasineur
            l_max_val = 0.0
            if is_emm and idx_emm_axis:
                l_max_val = calculer_L_max_draille(coords_fab, idx_emm_axis, nb_points)

            codes_choisis = {pt: st.session_state.get(f"acc_{pt}", "VOMC").split(" - ")[0] for pt in labels}
            L_max_lin = sum(longueurs_arcs_metres[b] for b in bords)
            list_longueurs_brutes = {b: mesures[b] for b in bords}
            # Génération du PDF (une seule fois — réutilisé pour ZIP et bouton PDF)
            _pdf_bytes = _gen_pdf(
                nom_client=nom_client or "—",
                reference=reference or "—",
                tissu=tissu,
                coloris=coloris or "—",
                date_str=date_aujourdhui,
                type_voile=type_voile,
                surf_m2=surf_m2,
                aire_pleine=aire_pleine,
                laize_choisie=laize_choisie,
                largeur_dispo=largeur_dispo,
                laize_a_couper=laize_a_couper,
                reste_final=reste_final,
                nb_laizes=nb_laizes,
                formule_bdf=formule_bdf,
                nb_bandes_rouleau=nb_bandes_rouleau,
                largeur_bdf_choisie=largeur_bdf_choisie,
                laize_utile_cm=laize_utile_cm,
                strategie=strategie,
                renforts_cm=renforts_cm_fab,
                labels=labels,
                longueurs_arcs_metres=longueurs_arcs_metres,
                perte=pertes_pct,
                long_rouleau_cm=MATERIAUX[tissu_select]["laizes"].get(laize_choisie, 50) * 100,
                coords_fab=coords_fab,
                courbures=courbures,
                bords=bords,
                nb_pts=nb_points,
                config_angles=config_angles,
                pts_emm=pts_emm,
                l_max=l_max_val,
                codes_accastillage=codes_choisis,
                list_pas=list_W,
                idx_cote_laize=idx_cote_laize,
                poids_surfacique=poids_surfacique,
                L_max_lin=L_max_lin,
                longueur_bdf_cm=longueur_bdf_cm,
                largeurs_bdf=largeurs_bdf,
                longueurs_bdf=longueurs_bdf,
                fixation_angles=fixation_angles,
                list_longueurs_brutes=list_longueurs_brutes,
                debit_total_atelier=surf_materiau_m2,
            )

            # Génération du ZIP (avec le PDF embarqué)
            _zip_complet = generer_zip_dxf(
                coords_fab=coords_fab,
                courbures=courbures,
                bords=bords,
                nb_pts=nb_points,
                labels=labels,
                renforts_cm=renforts_cm_fab,
                idx_cote_laize=idx_cote_laize,
                list_pas=list_pas,
                nom_client=nom_client or "TAROT",
                reference=reference,
                pdf_bytes=_pdf_bytes,
                tissu_select=tissu_select
            )

            c1.download_button(
                "📐 Export ZIP",
                data=_zip_complet,
                file_name=f"{'_'.join(filter(None, [nom_client, reference])) or 'tarot'}.zip",
                mime="application/zip",
                use_container_width=True
            )
            c2.download_button(
                "📄 Export PDF",
                data=_pdf_bytes,
                file_name=f"voile_{'_'.join(filter(None, [nom_client, reference])) or 'tarot'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Erreur lors de la génération des fichiers : {e}")
    else:
        c1.button("📐 Export ZIP", key="btn_zip", use_container_width=True, disabled=True)
        c2.button("📄 Export PDF", key="btn_pdf", use_container_width=True, disabled=True)

    with st.container(border=True):
        st.subheader("Fiche technique")
        if nom_client: st.markdown(f"👤 **{nom_client}**")
        st.markdown(f"Date : **{date_aujourdhui}**")
        st.markdown(f"**{tissu}** : {coloris}")
        if is_emm:
            st.markdown(f"**{type_voile} sur enrouleur**")
        else:
            st.markdown(f"**{type_voile}**")
        laize_a_couper = max(list_W) if list_W else 0
        reste_final    = largeur_dispo - laize_a_couper
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Laize rouleau", f"{laize_choisie} cm")
        c2.metric("Laize utile",    f"{largeur_dispo:.1f} cm")
        st.divider(); st.write("**Surfaces**")
        c1, c2 = st.columns(2)
        c1.metric("Surface réelle",   f"{surf_m2:.2f} m²")
        detail_bdf = f" + BDF {surf_bdf_saisie_m2:.2f} m²" if surf_bdf_saisie_m2 > 0 else " (BDF non saisi)"
        c2.metric("Surface matériau", f"{surf_materiau_m2:.2f} m²",
          help=f"Voile {surf_m2:.2f} + Renforts {surf_renforts_m2:.2f}{detail_bdf}")
        st.divider(); st.write("**Détail de la Coupe**")
        c1, c2 = st.columns(2)
        c1.metric("Nb de laizes",   f"{nb_laizes}")
        c2.metric("Laize à couper", f"{laize_a_couper:.1f} cm")
        (st.success if reste_final >= 0 else st.error)(
            f"{'✅ Reste' if reste_final >= 0 else '⚠️ Dépassement de'} "
            f"**{abs(reste_final):.1f} cm**{' sur rouleau' if reste_final >= 0 else ''}")
        if mode_economique and nb_laizes > 1:
            with st.expander("📏 Détail des laizes", expanded=True):
                for i, w in enumerate(list_W):
                    st.markdown(f"**Laize #{i+1}** : {w:.1f} cm")
        else:
            st.caption("💡 Toutes les laizes sont identiques.")
        with st.expander("🔍 Détail calcul"):
            st.write(f"Stratégie : **{strategie}**")
            st.write(f"BdF total : {formule_bdf}")
            st.write(f"Laize à couper : {laize_a_couper:.1f} cm "
                     f"({laize_a_couper - recouvr_actuel:.1f} visible + {recouvr_actuel} recouvrement)")
            # Cette ligne s'ajoutera uniquement si la laize est coupée en deux
            if division > 1:
                st.write(f"Mode de découpe : **1/2 laize active** (Laize utile : {laize_utile_cm:.1f} cm)")
        st.metric("Périmètre (L_max)", f"{L_max_lin:.2f} m")
        # Bilan matière
        st.divider()
        st.write("**Bilan Matière**")
        MARGE_CM = 8.0
        
        # if list_L:
        #     ml_cm = 0.0
        #     k = 0
        #     while k < nb_laizes:
        #         if k + 1 < nb_laizes:
        #             L_max = max(list_L[k], list_L[k + 1])
        #             L_min = min(list_L[k], list_L[k + 1])
        #             # Nesting réel : économie proportionnelle à l'effilement
        #             # → rectangulaire (L_min ≈ L_max) : ML ≈ L1+L2 (pas d'économie)
        #             # → triangulaire  (L_min ≈ 0)     : ML ≈ L_max (économie max)
        #             ml_cm += L_max + (L_min ** 2 / L_max) + MARGE_CM
        #             k += 2
        #         else:
        #             ml_cm += list_L[k] + MARGE_CM
        #             k += 1
        #     ml_m = ml_cm / 100
        
        longueur_tissu_utile = surf_materiau_m2 / ((laize_choisie - 2) / 100) if laize_choisie > 2 else 0
        st.metric(
            "📏 Longueur tissu utile",
            f"{longueur_tissu_utile:.2f} m",
            help="Surface matériau ÷ (laize − 2 cm)"
        )
        # --- CALCUL DU POIDS ---
        poids_surfacique = MATERIAUX[tissu_select]["poids"]
        masse_totale_kg = (surf_materiau_m2 * poids_surfacique) / 1000
        st.metric(
                label="Poids total estimé", 
                value=f"{masse_totale_kg:.2f} kg",
                help="Masse théorique incluant la voile, les renforts, les bordures et 10% de marge tissu."
            )
        st.write(f"({poids_surfacique} g/m²)")

        st.divider()
        st.write("**Détail des panneaux**")
        for i in range(len(list_L)):
            cp1, cp2, cp3 = st.columns([1,3,3])
            cp1.markdown(f"**#{i+1}**")
            cp2.markdown(f"↔️ {list_W[i]:.1f} cm")
            cp3.markdown(f"📏 {list_L[i]:.2f} cm")
        st.divider(); st.write("**Renforts par angle**")
        for i, pt in enumerate(LABELS[:nb_points]):
            if pt not in renforts_cm: continue
            rayons = renforts_cm[pt]
            couleur = COULEURS_RENFORT[i % len(COULEURS_RENFORT)]
        
            p_prev = coords_fab[(i-1) % nb_points]
            p_curr = coords_fab[i]
            p_next = coords_fab[(i+1) % nb_points]
            gx_disp = sum(p[0] for p in coords_fab) / nb_points
            gy_disp = sum(p[1] for p in coords_fab) / nb_points
        
            cp_in  = fleche_bezier(p_prev, p_curr, courbures[bords[(i-1) % nb_points]], gx_disp, gy_disp)[0]
            cp_out = fleche_bezier(p_curr, p_next, courbures[bords[i]],                 gx_disp, gy_disp)[0]
        
            def _ud(dx, dy):
                d = math.sqrt(dx**2 + dy**2)
                return (dx/d, dy/d) if d > 1e-9 else (1.0, 0.0)
        
            tv1 = _ud(cp_in[0]  - p_curr[0], cp_in[1]  - p_curr[1])
            tv2 = _ud(cp_out[0] - p_curr[0], cp_out[1] - p_curr[1])
            ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, tv1[0]*tv2[0] + tv1[1]*tv2[1]))))
        
            with st.expander(f"Angle {pt}' — {ang_deg:.1f}°"):
                for j, r in enumerate(rayons):
                    st.markdown(
                        f"<div style='background:rgba(205,92,92,0.1);color:#E9967A;"
                        f"padding:6px 14px;border-radius:5px;margin-bottom:4px;"
                        f"font-family:monospace;border-left:4px solid {couleur};'>"
                        f"Renfort {j+1} : <b>{r} cm</b></div>",
                        unsafe_allow_html=True)
        st.divider(); st.write("**Longueurs de courbure**")
        for b, dist_m in longueurs_arcs_metres.items():
            b_prime = "".join(c+"'" for c in b)
            st.markdown(
                f"<div style='background:rgba(205,92,92,0.1);color:#E9967A;padding:10px 18px;"
                f"border-radius:8px;margin-bottom:6px;font-family:monospace;"
                f"font-size:1.05em;border-left:5px solid #1a6fbf;'>"
                f"<b>{b_prime} :</b> {dist_m:.3f} m</div>",
                unsafe_allow_html=True)
            
        st.write("**Longueurs droites**")
        for i in range(nb_points):
            p1 = coords_fab[i]
            p2 = coords_fab[(i+1) % nb_points]
            label_seg = f"{LABELS[i]}'{LABELS[(i+1) % nb_points]}'"
            dist_m = longueur_cote(p1, p2) / 100
            st.markdown(
                f"<div style='background:rgba(26,111,191,0.08);color:#5ba3e0;padding:10px 18px;"
                f"border-radius:8px;margin-bottom:6px;font-family:monospace;"
                f"font-size:1.05em;border-left:5px solid #5ba3e0;'>"
                f"<b>{label_seg} :</b> {dist_m:.3f} m</div>",
                unsafe_allow_html=True)