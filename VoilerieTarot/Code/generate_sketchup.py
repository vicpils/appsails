# -*- coding: utf-8 -*-
"""Génère un fichier DXF pour import dans SketchUp.

Contient uniquement :
- Le contour fini de la voile (polyline fermée)
- Les lignes de séparation des laizes
Le tout dans un seul calque "VOILE".
"""
import math
import io
import ezdxf
import numpy as np


def generer_dxf_sketchup(
    coords_fab,
    coords_mur,
    courbures,
    bords,
    nb_pts,
    idx_cote_laize,
    list_W,
    recouvr_cm=2.5,
    nom_calque="VOILE",
):
    """Renvoie le DXF en bytes (contour + lignes de laizes, un seul calque).

    Args:
        coords_fab: liste de (x, y) — sommets de fabrication en cm.
        courbures: dict {bord_str: pct} — courbure de chaque côté en %.
        bords: liste des bords (ex. ["AB", "BC", "CA"]).
        nb_pts: nombre de sommets (3, 4 ou 5).
        idx_cote_laize: index du côté de référence pour l'orientation des laizes.
        list_W: largeurs des laizes en cm (incluant le recouvrement).
        recouvr_cm: recouvrement à soustraire pour obtenir la largeur visible.
        nom_calque: nom du calque unique.
    """
    doc = ezdxf.new("R2010")
    if nom_calque not in doc.layers:
        doc.layers.add(nom_calque, color=7)
    msp = doc.modelspace()

    # Centre de gravité — oriente les courbures vers l'extérieur
    gx = sum(p[0] for p in coords_fab) / nb_pts
    gy = sum(p[1] for p in coords_fab) / nb_pts

    def fleche_bezier_local(p1, p2, pct):
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(vx ** 2 + vy ** 2)
        if dist == 0:
            return mx, my
        nx, ny = -vy / dist, vx / dist
        if nx * (gx - mx) + ny * (gy - my) < 0:
            nx, ny = -nx, -ny
        f = (pct / 100) * dist
        return mx + nx * f * 2, my + ny * f * 2

    # 1. Contour de la voile — polyline fermée (Bézier discrétisée)
    contour_pts = []
    for i in range(nb_pts):
        p1 = coords_fab[i]
        p2 = coords_fab[(i + 1) % nb_pts]
        cp = fleche_bezier_local(p1, p2, courbures[bords[i]])
        for t in np.linspace(0, 1, 30, endpoint=False):
            x = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * cp[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * cp[1] + t ** 2 * p2[1]
            contour_pts.append((x, y))
    contour_pts.append(contour_pts[0])  # ferme la polyline
    msp.add_lwpolyline(contour_pts, close=True, dxfattribs={"layer": nom_calque})

    # 2. Lignes de séparation des laizes
    p1_ref = coords_fab[idx_cote_laize]
    p2_ref = coords_fab[(idx_cote_laize + 1) % nb_pts]
    angle = math.atan2(p2_ref[1] - p1_ref[1], p2_ref[0] - p1_ref[0])
    nx_l, ny_l = -math.sin(angle), math.cos(angle)
    dx_l, dy_l = math.cos(angle), math.sin(angle)

    projs = [p[0] * nx_l + p[1] * ny_l for p in contour_pts]
    d_min = min(projs)

    list_pas_cm = [w - recouvr_cm for w in list_W]

    cumul = 0.0
    for k in range(len(list_pas_cm) - 1):  # toutes les laizes sauf la dernière
        cumul += list_pas_cm[k]
        d_line = d_min + cumul

        intersections = []
        for j in range(len(contour_pts) - 1):
            d_j = contour_pts[j][0] * nx_l + contour_pts[j][1] * ny_l
            d_next = contour_pts[j + 1][0] * nx_l + contour_pts[j + 1][1] * ny_l
            if (d_j - d_line) * (d_next - d_line) < 0:
                t = (d_line - d_j) / (d_next - d_j)
                ix = contour_pts[j][0] + t * (contour_pts[j + 1][0] - contour_pts[j][0])
                iy = contour_pts[j][1] + t * (contour_pts[j + 1][1] - contour_pts[j][1])
                intersections.append((ix, iy))

        if len(intersections) >= 2:
            longs = sorted(intersections, key=lambda p: p[0] * dx_l + p[1] * dy_l)
            msp.add_line(longs[0], longs[-1], dxfattribs={"layer": nom_calque})
    # Traits de retrait à chaque angle (du mur jusqu'au point de fabrication)
    for i in range(nb_pts):
        msp.add_line(
            coords_mur[i],
            coords_fab[i],
            dxfattribs={"layer": nom_calque},
        )

    # 3. Sérialisation en bytes
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode("utf-8")