#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:56:50 2026
@author: jadeleroux
"""
"""
generate_dxf_tarot.py
=====================
Génère un fichier DXF par laize (panneau) de voile d'ombrage.

Convention de calques (alignée sur la machine du découpeur) :
  - Layer "1" : DÉCOUPE  — couleur ACI 10 (rouge)
  - Layer "8" : MARQUAGE — couleur ACI 50 (vert)

Selon le tissu, ce qui se trouve sur quel calque diffère :

  - Tissu standard (non-tentmesh) :
      * Contour de la laize  → "1" (découpe)
      * Numéro de la laize   → "8" (marquage)
      * Trait de recouvrement → "8" (marquage)

  - Tissu tentmesh (avec ourlet de 7 cm) :
      * Contour EXTÉRIEUR (offset 7 cm sur bords de voile, original sur
        coutures)                      → "1" (découpe)
      * Contour INTÉRIEUR (ligne de pli) → "8" (marquage)
      * Numéro de la laize              → "8" (marquage)
      * Trait de recouvrement           → "8" (marquage)

Pour les renforts, c'est plus simple : pas d'ourlet, donc le contour est
sur "1" et le numéro sur "8" quel que soit le tissu.
"""
import io
import math
import zipfile
import numpy as np
import ezdxf
# ── Constantes communes ────────────────────────────────────────────────────────
UNITE_DXF = 10
REPLI_TENTMESH_CM = 7.0   # marge de repli (cm) pour le tissu "tentmesh"
# ══════════════════════════════════════════════════════════════════════════════
# 1. GÉOMÉTRIE DE BÉZIER QUADRATIQUE
# ══════════════════════════════════════════════════════════════════════════════
def bezier_point(p1, cp, p2, t):
    """Évalue un point sur une courbe de Bézier quadratique."""
    return (
        (1-t)**2 * p1[0] + 2*(1-t)*t * cp[0] + t**2 * p2[0],
        (1-t)**2 * p1[1] + 2*(1-t)*t * cp[1] + t**2 * p2[1],
    )
def bezier_polyline(p1, cp, p2, n=64):
    """Discrétise la courbe de Bézier en n+1 points."""
    return [bezier_point(p1, cp, p2, t) for t in np.linspace(0, 1, n + 1)]
def get_control_point(p1, p2, pct, gx, gy):
    """
    Calcule le point de contrôle Bézier quadratique pour le bord p1→p2,
    avec une flèche de pct% de la longueur, côté intérieur (vers G).
    """
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    dist   = math.sqrt(vx**2 + vy**2)
    if dist == 0:
        return (mx, my)
    nx, ny = -vy / dist, vx / dist
    # La normale pointe vers le centre de gravité
    if nx * (gx - mx) + ny * (gy - my) < 0:
        nx, ny = -nx, -ny
    f = (pct / 100) * dist
    return (mx + nx * f * 2, my + ny * f * 2)
# ══════════════════════════════════════════════════════════════════════════════
# 1bis. TRACÉ DE REPLI (OFFSET LE LONG DES BORDS DE VOILE UNIQUEMENT)
# ══════════════════════════════════════════════════════════════════════════════
def _line_line_intersect(a1, a2, b1, b2):
    """Intersection de deux droites infinies (a1-a2) et (b1-b2)."""
    x1, y1 = a1; x2, y2 = a2
    x3, y3 = b1; x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
def build_repli_paths(contour_local, offset_dist, is_boundary=None):
    """
    Construit le tracé de repli pour une laize.
    Le repli ne suit que les bords de voile et IGNORE les arêtes de couture
    (entre deux laizes) — celles-ci seront cousues aux laizes voisines, pas
    repliées. À chaque transition bord-de-voile / couture, le tracé décalé
    est rattaché au sommet d'origine de la laize par un segment en
    bissectrice extérieure (jonction « miter » à 45° pour un coin droit),
    ce qui produit le pli propre attendu pour un ourlet.
    Paramètres
    ----------
    contour_local : list[(x, y)]
        Sommets de la laize en coordonnées locales (déjà à l'échelle DXF).
    offset_dist : float
        Distance d'offset extérieur, dans la même unité que contour_local.
    is_boundary : list[bool] | None
        Drapeau par sommet : True si le sommet est un point d'interpolation
        à la frontière de la laize (donc sur une couture), False s'il s'agit
        d'un point Bézier du bord de voile. Une arête est considérée comme
        couture si ses DEUX extrémités sont des points-frontière. Si None,
        on suppose qu'aucun sommet n'est sur une couture (mode rétro-compat).
    Retourne une liste de polylignes ouvertes (chacune une liste de (x, y)),
    une par tronçon de bord de voile.
    """
    n = len(contour_local)
    if n < 3 or offset_dist == 0:
        return []
    if is_boundary is None or len(is_boundary) != n:
        is_boundary = [False] * n
    def is_cut_edge(i):
        """L'arête (i → i+1) est une couture entre laizes."""
        return is_boundary[i] and is_boundary[(i + 1) % n]
    edges_cut = [is_cut_edge(i) for i in range(n)]
    # Cas dégénéré
    if all(edges_cut):
        return []
    # Orientation du polygone (formule du lacet)
    signed_area = 0.0
    for i in range(n):
        x1, y1 = contour_local[i]
        x2, y2 = contour_local[(i + 1) % n]
        signed_area += x1 * y2 - x2 * y1
    is_ccw = signed_area > 0
    def _unit(dx, dy):
        L = math.sqrt(dx * dx + dy * dy)
        return (dx / L, dy / L) if L > 1e-12 else (0.0, 0.0)
    def _outward_normal(p1, p2):
        """Normale unitaire extérieure de l'arête p1→p2 (selon orientation)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        L = math.sqrt(dx * dx + dy * dy)
        if L < 1e-12:
            return (0.0, 0.0)
        if is_ccw:
            return (dy / L, -dx / L)
        return (-dy / L, dx / L)
    def _miter(corner, d_p, n_p, d_c):
        """
        Point miter : intersection de la bissectrice extérieure en `corner`
        avec la droite de l'arête perimeter décalée.
          - d_p : direction unitaire du sommet vers l'arête perimeter
          - n_p : normale extérieure unitaire de l'arête perimeter
          - d_c : direction unitaire du sommet vers l'arête couture
        """
        bx = d_p[0] + d_c[0]
        by = d_p[1] + d_c[1]
        bn = math.sqrt(bx * bx + by * by)
        offset_start = (corner[0] + offset_dist * n_p[0],
                        corner[1] + offset_dist * n_p[1])
        if bn < 1e-9:
            # Arêtes opposées → pas vraiment de coin
            return offset_start
        # Bissectrice extérieure = -(d_p + d_c) normalisé
        ex_bx, ex_by = -bx / bn, -by / bn
        p1 = corner
        p2 = (corner[0] + ex_bx, corner[1] + ex_by)
        q1 = offset_start
        q2 = (offset_start[0] + d_p[0], offset_start[1] + d_p[1])
        pt = _line_line_intersect(p1, p2, q1, q2)
        return pt if pt is not None else offset_start
    # Aucune couture détectée : offset extérieur fermé classique
    if not any(edges_cut):
        # Polygone décalé complet (chaque sommet via intersection des arêtes
        # décalées consécutives). On retourne une polyligne fermée.
        offset_edges = []
        for i in range(n):
            p1 = contour_local[i]
            p2 = contour_local[(i + 1) % n]
            nx, ny = _outward_normal(p1, p2)
            offset_edges.append(((p1[0] + nx * offset_dist, p1[1] + ny * offset_dist),
                                 (p2[0] + nx * offset_dist, p2[1] + ny * offset_dist)))
        closed = []
        for i in range(n):
            prev_edge = offset_edges[(i - 1) % n]
            curr_edge = offset_edges[i]
            pt = _line_line_intersect(prev_edge[0], prev_edge[1],
                                      curr_edge[0], curr_edge[1])
            closed.append(pt if pt is not None else curr_edge[0])
        closed.append(closed[0])   # ferme explicitement
        return [closed]
    # Identifier les sous-chemins de perimeter (séquences contiguës d'arêtes non-couture)
    paths = []
    visited = [False] * n
    for start in range(n):
        if visited[start] or edges_cut[start]:
            continue
        if not edges_cut[(start - 1) % n]:
            # On n'est pas au début d'un sous-chemin → on attend
            continue
        edge_indices = []
        i = start
        guard = 0
        while not edges_cut[i] and guard < n + 1:
            edge_indices.append(i)
            visited[i] = True
            i = (i + 1) % n
            guard += 1
        # Sommets concernés
        vertex_indices = [edge_indices[0]] + [(idx + 1) % n for idx in edge_indices]
        m = len(vertex_indices)
        offset_pts = []
        for j in range(m):
            v_idx = vertex_indices[j]
            v = contour_local[v_idx]
            prev_v = contour_local[(v_idx - 1) % n]
            next_v = contour_local[(v_idx + 1) % n]
            if j == 0:
                # Transition couture → perimeter (début du sous-chemin)
                d_p = _unit(next_v[0] - v[0], next_v[1] - v[1])
                n_p = _outward_normal(v, next_v)
                d_c = _unit(prev_v[0] - v[0], prev_v[1] - v[1])
                offset_pts.append(_miter(v, d_p, n_p, d_c))
            elif j == m - 1:
                # Transition perimeter → couture (fin du sous-chemin)
                d_p = _unit(prev_v[0] - v[0], prev_v[1] - v[1])
                n_p = _outward_normal(prev_v, v)
                d_c = _unit(next_v[0] - v[0], next_v[1] - v[1])
                offset_pts.append(_miter(v, d_p, n_p, d_c))
            else:
                # Sommet intermédiaire : intersection des arêtes décalées voisines
                n_prev = _outward_normal(prev_v, v)
                n_curr = _outward_normal(v, next_v)
                e1a = (prev_v[0] + n_prev[0] * offset_dist,
                       prev_v[1] + n_prev[1] * offset_dist)
                e1b = (v[0] + n_prev[0] * offset_dist,
                       v[1] + n_prev[1] * offset_dist)
                e2a = (v[0] + n_curr[0] * offset_dist,
                       v[1] + n_curr[1] * offset_dist)
                e2b = (next_v[0] + n_curr[0] * offset_dist,
                       next_v[1] + n_curr[1] * offset_dist)
                pt = _line_line_intersect(e1a, e1b, e2a, e2b)
                offset_pts.append(pt if pt is not None else e1b)
        # Polyligne ouverte : sommet d'origine → bissectrice → offset → bissectrice → sommet d'origine
        start_corner = contour_local[vertex_indices[0]]
        end_corner = contour_local[vertex_indices[-1]]
        paths.append([start_corner] + offset_pts + [end_corner])
    return paths
# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSTRUCTION DU CONTOUR GLOBAL DE LA VOILE
#    Représenté comme une liste de segments (p1, cp, p2, type)
#    type = 'bezier' ou 'line'
# ══════════════════════════════════════════════════════════════════════════════
def build_sail_segments(coords_fab, courbures, bords, nb_pts):
    """
    Retourne une liste de dicts :
      { 'p1': ..., 'cp': ..., 'p2': ..., 'type': 'bezier'|'line' }
    Un dict par côté de la voile, dans l'ordre des bords.
    """
    gx = sum(p[0] for p in coords_fab) / nb_pts
    gy = sum(p[1] for p in coords_fab) / nb_pts
    segments = []
    for i in range(nb_pts):
        p1  = coords_fab[i]
        p2  = coords_fab[(i + 1) % nb_pts]
        b   = bords[i]
        pct = courbures.get(b, 0.0)
        cp  = get_control_point(p1, p2, pct, gx, gy)
        segments.append({'p1': p1, 'cp': cp, 'p2': p2, 'type': 'bezier'})
    return segments
def full_contour_polyline(segments, n_per_seg=128):
    """Discrétise tous les segments → liste de (x, y) sur le contour complet."""
    pts = []
    for seg in segments:
        sub = bezier_polyline(seg['p1'], seg['cp'], seg['p2'], n=n_per_seg)
        pts.extend(sub[:-1])   # évite la duplication du dernier point
    return pts
# ══════════════════════════════════════════════════════════════════════════════
# 3. SYSTÈME DE COORDONNÉES « LAIZE »
#    nx_l, ny_l : direction normale (largeur des laizes)
#    dx_l, dy_l : direction tangentielle (longueur des laizes)
# ══════════════════════════════════════════════════════════════════════════════
def laize_axes(coords_fab, idx_cote_laize, nb_pts):
    p1 = coords_fab[idx_cote_laize]
    p2 = coords_fab[(idx_cote_laize + 1) % nb_pts]
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    nx_l  = -math.sin(angle)
    ny_l  =  math.cos(angle)
    dx_l  =  math.cos(angle)
    dy_l  =  math.sin(angle)
    return nx_l, ny_l, dx_l, dy_l
# ══════════════════════════════════════════════════════════════════════════════
# 4. INTERSECTION D'UNE POLYLIGNE AVEC UNE DROITE PARALLÈLE AUX LAIZES
#    La droite est définie par : proj_normale == d_cut
#    où proj_normale = x * nx_l + y * ny_l
# ══════════════════════════════════════════════════════════════════════════════
def intersect_polyline_at(pts_contour, nx_l, ny_l, d_cut, n_max=2):
    """
    Cherche les intersections de la polyligne avec la droite proj == d_cut.
    Retourne jusqu'à n_max points d'intersection (x, y).
    Tri selon la projection tangentielle.
    """
    results = []
    projs   = [p[0] * nx_l + p[1] * ny_l for p in pts_contour]
    for i in range(len(pts_contour) - 1):
        d0, d1 = projs[i], projs[i + 1]
        if (d0 - d_cut) * (d1 - d_cut) <= 0 and d0 != d1:
            alpha = (d_cut - d0) / (d1 - d0)
            x = pts_contour[i][0] + alpha * (pts_contour[i+1][0] - pts_contour[i][0])
            y = pts_contour[i][1] + alpha * (pts_contour[i+1][1] - pts_contour[i][1])
            results.append((x, y))
    # Dédoublonnage (points très proches)
    dedup = []
    for p in results:
        if not any(math.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2) < 1e-3 for q in dedup):
            dedup.append(p)
    return dedup[:n_max]
# ══════════════════════════════════════════════════════════════════════════════
# 5. EXTRACTION D'UNE LAIZE
#    Retourne le contour fermé de la laize k (liste de (x,y))
#    en coupant le contour global entre d_start et d_end,
#    ET la liste is_boundary[i] indiquant si le point i est un point
#    d'interpolation sur la frontière (donc sur une couture entre laizes).
# ══════════════════════════════════════════════════════════════════════════════
def extract_laize_contour(pts_contour_full, nx_l, ny_l, dx_l, dy_l,
                          d_start, d_end):
    """
    Extrait le contour ordonné de la laize entre d_start et d_end.
    Retourne (contour_local, is_boundary) :
      - contour_local : liste de (x, y) en coordonnées locales
        (x = longueur côté réf., y = largeur laize ramenée à 0 en bas)
      - is_boundary   : liste de bool, même longueur que contour_local.
                        True si le point a été inséré par interpolation à
                        la frontière (donc situé sur la couture séparant
                        cette laize d'une voisine), False si le point est
                        un point bézier du bord de voile.
    Cette information permet à write_laize_dxf / build_repli_paths de
    distinguer EXACTEMENT (sans tolérance numérique) les arêtes de
    couture des arêtes de bord de voile : une arête est une couture si
    ET SEULEMENT SI ses deux extrémités sont des points-frontière.
    """
    EPS = 1e-9
    n   = len(pts_contour_full)
    projs = [p[0] * nx_l + p[1] * ny_l for p in pts_contour_full]
    def interp(i, d_thresh):
        p1 = pts_contour_full[i]
        p2 = pts_contour_full[(i + 1) % n]
        d1, d2 = projs[i], projs[(i + 1) % n]
        if abs(d2 - d1) < EPS:
            return p1
        alpha = (d_thresh - d1) / (d2 - d1)
        return (p1[0] + alpha * (p2[0] - p1[0]),
                p1[1] + alpha * (p2[1] - p1[1]))
    def in_band(d):
        return d_start - EPS <= d <= d_end + EPS
    # Chaque segment est une liste de tuples (point, is_boundary)
    segments_in_band = []
    cur = None
    for i in range(n):
        dc = projs[i]
        dn = projs[(i + 1) % n]
        # Entrée dans la bande : commencer un segment avec le point d'intersection
        if not in_band(dc) and in_band(dn):
            cur = []
            if dc < d_start:
                cur.append((interp(i, d_start), True))
            elif dc > d_end:
                cur.append((interp(i, d_end), True))
        # Point dans la bande : l'ajouter au segment courant
        if in_band(dc):
            if cur is None:
                cur = []
            cur.append((pts_contour_full[i], False))
        # Sortie de la bande : clore le segment avec le point d'intersection
        if in_band(dc) and not in_band(dn):
            if cur is not None:
                if dn > d_end:
                    cur.append((interp(i, d_end), True))
                elif dn < d_start:
                    cur.append((interp(i, d_start), True))
                segments_in_band.append(cur)
                cur = None
    # Fin de boucle dans la bande (laize de bord finale)
    if cur:
        segments_in_band.append(cur)
    if not segments_in_band:
        return [], []
    if len(segments_in_band) == 1:
        # Laize de bord : un seul passage continu
        pts_with_flags = segments_in_band[0]
    else:
        # Cas normal : deux passages, assembler dans l'ordre naturel (PAS de reversed)
        # seg[0] va de d_start→d_end, seg[1] va de d_end→d_start
        # → seg[0] + seg[1] forme un polygone fermé correct
        pts_with_flags = segments_in_band[0] + segments_in_band[1]
    pts_raw     = [pf[0] for pf in pts_with_flags]
    is_boundary = [pf[1] for pf in pts_with_flags]
    # Passage en coordonnées locales
    xs_loc = [p[0] * dx_l + p[1] * dy_l for p in pts_raw]
    ys_loc = [p[0] * nx_l + p[1] * ny_l for p in pts_raw]
    x0 = min(xs_loc)
    y0 = d_start
    contour_local = [(x - x0, y - y0) for x, y in zip(xs_loc, ys_loc)]
    return contour_local, is_boundary
def _combine_repli_paths_to_outer_polygon(repli_paths):
    """
    Concatène les segments de repli retournés par build_repli_paths()
    en un seul polygone fermé représentant le contour extérieur de la
    laize (avec offset 7 cm sur les bords de voile).

    - Sans couture (laize unique) : build_repli_paths retourne déjà un
      polygone fermé ; on retire simplement le point dupliqué de
      fermeture pour pouvoir l'utiliser avec close=True.
    - Avec coutures : chaque segment de repli va d'un sommet d'origine
      à un autre, et la jonction entre deux segments consécutifs
      correspond exactement à une arête de couture (non offset). En
      concaténant les segments dans leur ordre naturel, les arêtes
      implicites entre eux reconstituent les coutures.

    Retourne une liste de (x, y), à utiliser avec close=True.
    """
    if not repli_paths:
        return []
    outer = []
    for path in repli_paths:
        if not path or len(path) < 2:
            continue
        # Si le path est déjà fermé explicitement (cas sans couture), on
        # retire le dernier point dupliqué.
        same_first_last = (
            abs(path[0][0] - path[-1][0]) < 1e-6 and
            abs(path[0][1] - path[-1][1]) < 1e-6
        )
        if same_first_last:
            outer.extend(path[:-1])
        else:
            outer.extend(path)
    return outer
def write_laize_dxf(contour_local, laize_num, client_name="TAROT", text_h=40.0, margin=15.0, recouvr_pts=None, repli_paths=None, y_min_is_cut=True, y_max_is_cut=True, is_boundary=None):
    """
    Génère le DXF d'une laize, aligné sur la convention de calques du
    dessinateur :
      - Calque "1" (couleur 10) : DÉCOUPE
      - Calque "8" (couleur 50) : MARQUAGE (numéro, recouvrement, pli)

    En tissu STANDARD :
      * Le contour de la laize est sur "1" (découpe).
    En tissu TENTMESH (si repli_paths est fourni) :
      * Le contour intérieur (= ligne de pli) est sur "8" (marquage).
      * Le contour extérieur (offset 7 cm sur les bords de voile, identique
        à l'intérieur sur les coutures) est sur "1" (découpe).
    """
    doc = ezdxf.new(dxfversion="R2010")
    doc.header['$INSUNITS'] = 4  # Millimètres
    msp = doc.modelspace()

    # --- ALIGNEMENT STRICT SUR LES CALQUES DU DESSINATEUR ---
    doc.layers.add("1", color=10)  # Découpe
    doc.layers.add("8", color=50)  # Marquages et textes internes

    is_tentmesh = bool(repli_paths)
    # En tentmesh, le contour intérieur n'est plus la découpe : c'est la
    # ligne de pli, donc un marquage. La découpe se fait sur le contour
    # extérieur (offset 7 cm sur bords de voile).
    inner_layer = "8" if is_tentmesh else "1"

    # 1. Tracé du contour intérieur
    n_c = len(contour_local)
    if n_c >= 2:
        pts_3d = [(x, y, 0.0) for x, y in contour_local]
        msp.add_lwpolyline(pts_3d, close=True, dxfattribs={"layer": inner_layer})

    # 1bis. Pour tentmesh : tracé du contour EXTÉRIEUR sur "1" (découpe)
    if is_tentmesh:
        outer = _combine_repli_paths_to_outer_polygon(repli_paths)
        if outer and len(outer) >= 3:
            pts_3d = [(x, y, 0.0) for x, y in outer]
            msp.add_lwpolyline(pts_3d, close=True, dxfattribs={"layer": "1"})
    # 2. Trait de recouvrement INTERNE (Sur le calque "8")
    if recouvr_pts and len(recouvr_pts) >= 2:
        msp.add_line(recouvr_pts[0], recouvr_pts[1], dxfattribs={"layer": "8"})
    # 3. Placement du numéro de panneau (Sur le calque "8")
    if contour_local:
        ys = [p[1] for p in contour_local]
        y_min = min(ys)
        y_max = max(ys)
        panel_height = y_max - y_min

        def get_x_bounds(y_level, contour, marge):
            inter = []
            n = len(contour)
            for i in range(n):
                p1, p2 = contour[i], contour[(i+1) % n]
                if (p1[1] - y_level) * (p2[1] - y_level) <= 0 and p1[1] != p2[1]:
                    alpha = (y_level - p1[1]) / (p2[1] - p1[1])
                    inter.append(p1[0] + alpha * (p2[0] - p1[0]))
            if len(inter) >= 2:
                inter.sort()
                return inter[0] + marge, inter[-1] - marge
            return None
        is_last_panel = not y_max_is_cut
        if is_last_panel:
            text_h_eff = 10.0
            margin_eff = 1.0
            y_base_texte = y_max - text_h_eff - 2.0
        else:
            text_h_eff = min(text_h, panel_height * 0.40)
            margin_eff = min(margin, panel_height * 0.15)
            if recouvr_pts:
                rec_y = recouvr_pts[0][1]
                dist_to_top, dist_to_bottom = y_max - rec_y, rec_y - y_min
                if dist_to_top <= dist_to_bottom:
                    y_base_texte = (rec_y + y_max) / 2 - (text_h_eff / 2)
                else:
                    y_base_texte = (y_min + rec_y) / 2 - (text_h_eff / 2)
            else:
                y_base_texte = y_min + margin_eff
        nb_chars = len(str(laize_num))
        largeur_estimee = text_h_eff * 0.7 * nb_chars
        bounds = get_x_bounds(y_base_texte + text_h_eff/2, contour_local, margin_eff)
        if bounds:
            x_min_safe, x_max_safe = bounds
        else:
            xs = [p[0] for p in contour_local]
            x_min_safe, x_max_safe = min(xs) + margin_eff, max(xs) - margin_eff
        tx = x_max_safe - largeur_estimee
        ty = y_base_texte
        # Ajout du texte sur le calque "8"
        msp.add_text(
            str(laize_num),
            dxfattribs={
                "layer": "8",
                "height": text_h_eff,
                "insert": (tx, ty),
            }
        )
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")
# ══════════════════════════════════════════════════════════════════════════════
# 8. DXF D'UN RENFORT
# ══════════════════════════════════════════════════════════════════════════════
def _bezier_sub_to_radius(p_start, cp, p_end, rayon, n=128):
    """
    Points de la Bézier quadratique (p_start→cp→p_end) depuis t=0
    jusqu'à la première fois où la distance à p_start atteint rayon.
    """
    pts = [p_start]
    px, py, pd = p_start[0], p_start[1], 0.0
    for k in range(1, n + 1):
        t = k / n
        x = (1-t)**2*p_start[0] + 2*(1-t)*t*cp[0] + t**2*p_end[0]
        y = (1-t)**2*p_start[1] + 2*(1-t)*t*cp[1] + t**2*p_end[1]
        d = math.sqrt((x - p_start[0])**2 + (y - p_start[1])**2)
        if d >= rayon:
            if d > pd:
                alpha = (rayon - pd) / (d - pd)
                pts.append((px + alpha*(x - px), py + alpha*(y - py)))
            return pts
        pts.append((x, y))
        px, py, pd = x, y, d
    return pts


def write_renfort_dxf(centre, rayon_cm, angle_sommet_deg, v1, v2,
                      renfort_num, nom_client="TAROT",
                      rayon_min=None, centre_gravite=None,
                      p_prev_mm=None, cp_in_mm=None,
                      p_next_mm=None, cp_out_mm=None):
    """
    Génère le DXF d'un renfort, aligné sur la convention de calques du
    dessinateur :
      - Calque "1" (couleur 10) : contour à découper (arc + 2 segments
                                    droits vers le sommet)
      - Calque "8" (couleur 50) : numéro de repère (marquage interne)
    """
    doc = ezdxf.new(dxfversion="R2010")
    doc.header['$INSUNITS'] = 4  # Millimètres
    msp = doc.modelspace()
    # Même convention de calques que les laizes
    doc.layers.add("1", color=10)  # Découpe
    doc.layers.add("8", color=50)  # Marquage / texte
    cx, cy = centre
    
    # ── ALIGNEMENT DROIT FIL ─────────────────────────────────────────────────
    _bx0 = v1[0] + v2[0]
    _by0 = v1[1] + v2[1]
    _bn0 = math.sqrt(_bx0**2 + _by0**2)
    if _bn0 > 1e-9:
        _bx0, _by0 = _bx0 / _bn0, _by0 / _bn0
    else:
        _bx0, _by0 = 0.0, 1.0
    # Orienter la bissectrice vers l'intérieur de la voile
    if centre_gravite is not None:
        _gx0, _gy0 = centre_gravite
        if _bx0 * (_gx0 - cx) + _by0 * (_gy0 - cy) < 0:
            _bx0, _by0 = -_bx0, -_by0
    # Angle à faire tourner pour aligner biss. sur +Y  (π/2 − atan2(by, bx))
    _alpha = math.atan2(_by0, _bx0)
    _rot   = math.pi / 2.0 - _alpha
    _cr, _sr = math.cos(_rot), math.sin(_rot)
    _ox, _oy = cx, cy   # centre original avant réinitialisation
 
    def _rp(px, py):
        """Tourne (px, py) autour du centre original → repère droit fil."""
        dx, dy = px - _ox, py - _oy
        return (dx * _cr - dy * _sr, dx * _sr + dy * _cr)
 
    def _rv(vx, vy):
        """Tourne un vecteur libre."""
        return (vx * _cr - vy * _sr, vx * _sr + vy * _cr)
 
    # Appliquer la rotation à tous les paramètres
    cx, cy = 0.0, 0.0
    v1 = _rv(v1[0], v1[1])
    v2 = _rv(v2[0], v2[1])
    if centre_gravite is not None:
        centre_gravite = _rp(centre_gravite[0], centre_gravite[1])
    if p_prev_mm is not None:
        p_prev_mm = _rp(p_prev_mm[0], p_prev_mm[1])
    if cp_in_mm is not None:
        cp_in_mm  = _rp(cp_in_mm[0],  cp_in_mm[1])
    if p_next_mm is not None:
        p_next_mm = _rp(p_next_mm[0], p_next_mm[1])
    if cp_out_mm is not None:
        cp_out_mm = _rp(cp_out_mm[0], cp_out_mm[1])
    # ─────────────────────────────────────────────────────────────────────────
    
    # ── Angles de l'arc ───────────────────────────────────────────────────────
    a1_deg = math.degrees(math.atan2(v1[1], v1[0])) % 360
    a2_deg = math.degrees(math.atan2(v2[1], v2[0])) % 360
    # Choisir le sens de l'arc qui couvre l'intérieur de l'angle.
    # On utilise la bissectrice intérieure (déjà calculée) pour valider :
    # le milieu de l'arc correct doit pointer dans la même direction que la bissectrice.
    # Bissectrice intérieure (recalculée ici pour être sûr)
    _bx = v1[0] + v2[0]; _by = v1[1] + v2[1]
    _bn = math.sqrt(_bx**2 + _by**2)
    if _bn > 1e-9:
        _bx, _by = _bx/_bn, _by/_bn
    if centre_gravite is not None:
        _gx, _gy = centre_gravite
        if _bx*(_gx-cx) + _by*(_gy-cy) < 0:
            _bx, _by = -_bx, -_by
    # Angle de la bissectrice intérieure
    bis_deg = math.degrees(math.atan2(_by, _bx)) % 360
    # L'arc CCW de a1→a2 : son milieu est à a1 + ccw_span/2
    ccw_span = (a2_deg - a1_deg) % 360
    mid_ccw  = (a1_deg + ccw_span / 2) % 360
    # L'arc CW de a1→a2 (= CCW de a2→a1) : son milieu est à a2 + (360-ccw_span)/2
    cw_span  = 360 - ccw_span
    mid_cw   = (a2_deg + cw_span / 2) % 360
    # Choisir l'arc dont le milieu est le plus proche de la bissectrice
    def angle_diff(a, b):
        return min((a - b) % 360, (b - a) % 360)
    if angle_diff(mid_ccw, bis_deg) <= angle_diff(mid_cw, bis_deg):
        arc_start, arc_end = a1_deg, a2_deg   # CCW
    else:
        arc_start, arc_end = a2_deg, a1_deg   # CCW dans l'autre sens
    # ── Secteur fermé : arc + 2 segments droits vers le sommet ────────────────
    # Points extrémités de l'arc
    p_arc1 = (cx + rayon_cm * math.cos(math.radians(arc_start)),
              cy + rayon_cm * math.sin(math.radians(arc_start)))
    p_arc2 = (cx + rayon_cm * math.cos(math.radians(arc_end)),
              cy + rayon_cm * math.sin(math.radians(arc_end)))
    # msp.add_arc(
    #     center=(cx, cy, 0),
    #     radius=rayon_cm,
    #     start_angle=arc_start,
    #     end_angle=arc_end,
    #     dxfattribs={"layer": "1"}
    # )
    if cp_in_mm is not None and p_prev_mm is not None:
        side1 = _bezier_sub_to_radius((cx, cy), cp_in_mm, p_prev_mm, rayon_cm)
        end1 = side1[-1]
    else:
        end1 = p_arc1
        side1 = None
    
    if cp_out_mm is not None and p_next_mm is not None:
        side2 = _bezier_sub_to_radius((cx, cy), cp_out_mm, p_next_mm, rayon_cm)
        end2 = side2[-1]
    else:
        end2 = p_arc2
        side2 = None
    
    # Arc recalculé depuis les vrais endpoints des Bézier
    arc_start_real = math.degrees(math.atan2(end1[1] - cy, end1[0] - cx)) % 360
    arc_end_real   = math.degrees(math.atan2(end2[1] - cy, end2[0] - cx)) % 360
    ccw_span_r = (arc_end_real - arc_start_real) % 360
    mid_ccw_r  = (arc_start_real + ccw_span_r / 2) % 360
    cw_span_r  = 360 - ccw_span_r
    mid_cw_r   = (arc_end_real + cw_span_r / 2) % 360
    
    if angle_diff(mid_ccw_r, bis_deg) > angle_diff(mid_cw_r, bis_deg):
        arc_start_real, arc_end_real = arc_end_real, arc_start_real
    msp.add_arc(
        center=(cx, cy, 0),
        radius=rayon_cm,
        start_angle=arc_start_real,
        end_angle=arc_end_real,
        dxfattribs={"layer": "1"}
    )
    
    # Tracé des côtés
    if side1:
        msp.add_lwpolyline([(x, y, 0) for x, y in side1], dxfattribs={"layer": "1"})
    else:
        msp.add_line((cx, cy, 0), (end1[0], end1[1], 0), dxfattribs={"layer": "1"})
    
    if side2:
        msp.add_lwpolyline([(x, y, 0) for x, y in side2], dxfattribs={"layer": "1"})
    else:
        msp.add_line((cx, cy, 0), (end2[0], end2[1], 0), dxfattribs={"layer": "1"})
    # --- PLACEMENT DU NUMÉRO ---
    # On calcule la bissectrice intérieure (bx, by)
    bx = v1[0] + v2[0]
    by = v1[1] + v2[1]
    bn = math.sqrt(bx**2 + by**2)
    if bn < 1e-9:
        bx, by = -v1[1], v1[0]
        bn = 1.0
    bx, by = bx/bn, by/bn
    # Inversion si elle pointe vers l'extérieur
    if centre_gravite is not None:
        to_gx, to_gy = centre_gravite[0] - cx, centre_gravite[1] - cy
        if bx * to_gx + by * to_gy < 0:
            bx, by = -bx, -by
    ## --- PLACEMENT DYNAMIQUE DU NUMÉRO (rotation + bornes géométriques) ---
    nb_chars = len(str(renfort_num))
    ## --- PLACEMENT À 2 CM DU SOMMET ---
    R_TEXTE = 20.0   # mm, soit 2 cm — distance imposée
    H_TEXTE = 10.0
    nb_chars = len(str(renfort_num))
    # Demi-angle au sommet
    demi_angle_rad = math.radians(angle_sommet_deg / 2.0)
    tan_demi = math.tan(demi_angle_rad)
    MARGE_LAT = 3.0   # garde latérale (mm)
    MARGE_ARC = 3.0   # garde par rapport à l'arc (mm)
    # Hauteur de texte "idéale" si l'angle est confortable
    H_IDEALE = 10.0
    CHAR_W    = 0.85
    SECURITE  = 0.60
    # Hauteur max acceptable pour que le texte tienne entre les 2 côtés à R_TEXTE
    if tan_demi > 1e-6:
        denom     = 1.0 + nb_chars * CHAR_W * tan_demi
        h_max_lat = 2.0 * (R_TEXTE * tan_demi - MARGE_LAT) / denom
    else:
        h_max_lat = H_IDEALE
    # Hauteur max acceptable pour que la longueur du texte ne dépasse pas l'arc
    # (texte centré sur R_TEXTE, longueur ≈ nb_chars * 0.7 * H)
    longueur_dispo = 2.0 * (rayon_cm - R_TEXTE - MARGE_ARC)
    if longueur_dispo > 0:
        h_max_rad = longueur_dispo / (nb_chars * 0.7)
    else:
        h_max_rad = H_IDEALE
    H_TEXTE = min(H_IDEALE, h_max_lat * SECURITE, h_max_rad * SECURITE)
    H_TEXTE = max(3.0, H_TEXTE)
    tx = cx + bx * R_TEXTE
    ty = cy + by * R_TEXTE
    # Rotation du texte le long de la bissectrice (en gardant le chiffre "à l'endroit")
    angle_text_deg = math.degrees(math.atan2(by, bx))
    if angle_text_deg > 90:
        angle_text_deg -= 180
    elif angle_text_deg < -90:
        angle_text_deg += 180
    t = msp.add_text(
        str(renfort_num),
        dxfattribs={
            "layer": "8",
            "height": H_TEXTE,
            "rotation": angle_text_deg,
        }
    )
    t.set_placement((tx, ty), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")
# ══════════════════════════════════════════════════════════════════════════════
# 7. POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def generer_zip_dxf(coords_fab, courbures, bords, nb_pts, labels,
                    renforts_cm, idx_cote_laize, list_pas,
                    nom_client="TAROT", pdf_bytes=None, reference="",
                    tissu_select="standard"):
    # 1. Contour global
    segments = build_sail_segments(coords_fab, courbures, bords, nb_pts)
    pts_full = full_contour_polyline(segments, n_per_seg=256)
    # 2. Axes laize
    nx_l, ny_l, dx_l, dy_l = laize_axes(coords_fab, idx_cote_laize, nb_pts)
    # 3. Largeur totale réelle (cm)
    projs = [p[0] * nx_l + p[1] * ny_l for p in pts_full]
    d_min, d_max = min(projs), max(projs)
    largeur_totale_cm = d_max - d_min
    # --- CORRECTION CRITIQUE DU NOMBRE DE LAIZES ---
    # En mode économique, list_pas contient déjà les valeurs réelles proportionnelles à l'échelle 's'.
    # On applique le ratio pour retrouver les centimètres exacts, sans écraser les laizes asymétriques.
    sum_pas = sum(list_pas)
    if sum_pas > 0:
        list_pas_cm = [float(p) * (largeur_totale_cm / sum_pas) for p in list_pas]
    else:
        list_pas_cm = [largeur_totale_cm]
    is_tentmesh = (str(tissu_select).strip().lower() == "tentmesh")
    repli_offset_mm = REPLI_TENTMESH_CM * UNITE_DXF if is_tentmesh else 0.0
    RECOUVR_CM = 3.0 if is_tentmesh else 2.5
    text_h_cm = RECOUVR_CM * 0.8
    margin_cm = text_h_cm * 1.5
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        cumul = 0.0
        # Parcours de TOUTES les laizes calculées
        for k, pas_visible in enumerate(list_pas_cm):
            d_start = d_min + cumul
            # --- LOGIQUE DE COUPE ET RECOUVREMENT ASSOCIE ---
            # La laize k va de d_start à d_end_coupe (qui inclut le recouvrement si ce n'est pas la dernière)
            if k < len(list_pas_cm) - 1:
                d_end_coupe = d_start + pas_visible + RECOUVR_CM
            else:
                d_end_coupe = d_max # Force la dernière borne pour éviter les sauts de décimales
            # Extraction du contour géométrique exact + marquage des points-frontière
            contour_cm, is_boundary = extract_laize_contour(
                pts_full, nx_l, ny_l, dx_l, dy_l, d_start, d_end_coupe
            )
            # On fait avancer le cumul du pas visible pour la laize suivante
            cumul += pas_visible
            if not contour_cm:
                continue
            # Mise à l'échelle DXF (cm -> mm) — is_boundary suit le même indexage
            contour_mm = [(x * UNITE_DXF, y * UNITE_DXF) for x, y in contour_cm]
            text_h_mm = text_h_cm * UNITE_DXF
            margin_mm = margin_cm * UNITE_DXF
            y_min_is_cut = (k > 0)
            y_max_is_cut = (k < len(list_pas_cm) - 1)
            # Repli spécifique Tentmesh
            repli_paths_mm = None
            if is_tentmesh and repli_offset_mm > 0:
                repli_paths_mm = build_repli_paths(
                    contour_mm, repli_offset_mm,
                    is_boundary=is_boundary,
                )
            # --- CORRECTION DU TRAIT DE RECOUVREMENT INTERNE ---
            # Le trait doit marquer la fin du pas visible (d_start + pas_visible)
            # à l'intérieur de la laize découper, pour guider la superposition.
            recouvr_pts_mm = None
            if k < len(list_pas_cm) - 1:
                d_repere = d_start + pas_visible
                # On cherche les intersections à la frontière exacte du pas visible
                crossings_r = intersect_polyline_at(pts_full, nx_l, ny_l, d_repere, n_max=2)
                if len(crossings_r) >= 2:
                    # Calcul de l'origine locale X pour cette laize
                    pts_in_band = [p for p in pts_full if d_start - 1e-9 <= p[0]*nx_l+p[1]*ny_l <= d_end_coupe + 1e-9]
                    x0_orig = min(p[0]*dx_l+p[1]*dy_l for p in pts_in_band) if pts_in_band else 0.0
                    recouvr_pts_mm = [
                        ((p[0]*dx_l + p[1]*dy_l - x0_orig) * UNITE_DXF,
                         (p[0]*nx_l + p[1]*ny_l - d_start) * UNITE_DXF)
                        for p in crossings_r
                    ]
            # Écriture du fichier DXF individuel
            dxf_bytes = write_laize_dxf(contour_mm, k+1, nom_client,
                                        text_h_mm, margin_mm,
                                        recouvr_pts_mm, repli_paths_mm,
                                        y_min_is_cut=y_min_is_cut,
                                        y_max_is_cut=y_max_is_cut,
                                        is_boundary=is_boundary)
            zf.writestr(f"laize_{k+1:02d}_{nom_client}_{reference}.dxf", dxf_bytes)
        # ── Génération des Renforts ──────────────────────────────────────────
        renfort_num = len(list_pas_cm) + 1
        gx_cm = sum(p[0] for p in coords_fab)/nb_pts
        gy_cm = sum(p[1] for p in coords_fab)/nb_pts
        gravite_mm = (gx_cm * UNITE_DXF, gy_cm * UNITE_DXF)
        
        N_SEG    = 256          # doit correspondre au n_per_seg de full_contour_polyline
        n_total  = len(pts_full)
        STEP_TANGENTE = 22
        
        for i in range(nb_pts):
            label = labels[i]
            if label in renforts_cm:
                p_curr = coords_fab[i]
                N_SEG = 256
                idx_v    = i * N_SEG
                n_total  = len(pts_full)
                STEP_TANGENTE = 12
                p_before = pts_full[(idx_v - STEP_TANGENTE) % n_total]
                p_after  = pts_full[(idx_v + STEP_TANGENTE) % n_total]
        
                def _u(dx, dy):
                    d = math.sqrt(dx**2 + dy**2)
                    return (dx/d, dy/d) if d > 1e-9 else (1.0, 0.0)
        
                v1 = _u(p_before[0] - p_curr[0], p_before[1] - p_curr[1])
                v2 = _u(p_after[0]  - p_curr[0], p_after[1]  - p_curr[1])
                angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, v1[0]*v2[0]+v1[1]*v2[1]))))
                p_curr_mm = (p_curr[0] * UNITE_DXF, p_curr[1] * UNITE_DXF)
        
                # ← NOUVEAU : points de contrôle Bézier des deux côtés adjacents
                seg_avant = segments[(i - 1) % nb_pts]
                seg_apres = segments[i]
                cp_in_mm  = (seg_avant['cp'][0] * UNITE_DXF, seg_avant['cp'][1] * UNITE_DXF)
                cp_out_mm = (seg_apres['cp'][0] * UNITE_DXF, seg_apres['cp'][1] * UNITE_DXF)
                p_prev_mm = (seg_avant['p1'][0] * UNITE_DXF, seg_avant['p1'][1] * UNITE_DXF)
                p_next_mm = (seg_apres['p2'][0] * UNITE_DXF, seg_apres['p2'][1] * UNITE_DXF)
        
                for rayon_cm in renforts_cm[label]:
                    rayon_mm = rayon_cm * UNITE_DXF
                    dxf_renfort = write_renfort_dxf(        # ← appel remplacé
                        p_curr_mm, rayon_mm, angle_deg, v1, v2,
                        renfort_num, nom_client, centre_gravite=gravite_mm,
                        p_prev_mm=p_prev_mm, cp_in_mm=cp_in_mm,
                        p_next_mm=p_next_mm, cp_out_mm=cp_out_mm,
                    )
                    zf.writestr(
                        f"renfort_{renfort_num:02d}_{label}_{nom_client}_{reference}.dxf",
                        dxf_renfort
                    )
                    renfort_num += 1
        if pdf_bytes is not None:
            filename_pdf = f"fiche_technique_{nom_client}_{reference}.pdf"
            zf.writestr(filename_pdf, pdf_bytes)
    zip_buf.seek(0)
    return zip_buf.read()