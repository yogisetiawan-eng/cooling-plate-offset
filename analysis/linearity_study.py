"""How linear is offset vs room temperature, using published correlations.

Convection: Churchill & Chu (1975) laminar vertical-plate correlation, with
g*cos(theta) for an inclined plate (up to 60 deg from vertical); McAdams
Nu = 0.54 Ra^(1/4) and 0.27 Ra^(1/4) for the two faces of a horizontal plate.
Air properties: evaluated at film temperature, interpolated from standard
1 atm values (Incropera tables).
Radiation: grey plate in large surroundings at room temperature.

Plate size, angle and finish are unknown, so all are swept.
"""
import numpy as np

SIGMA = 5.670e-8
SP, CHAMBER = 34.5, 37.0
RT, A_MEAS = (22.0, 17.5), (35.5, 39.75)

# T_K, k W/mK, nu m2/s, alpha m2/s
AIR = np.array([[250, .0223, 11.44e-6, 15.9e-6],
                [300, .0263, 15.89e-6, 22.5e-6],
                [350, .0300, 20.92e-6, 29.9e-6]])


def props(t_c):
    t = t_c + 273.15
    return tuple(np.interp(t, AIR[:, 0], AIR[:, i]) for i in (1, 2, 3))


def h_conv(ts, ta, L, orient):
    tf = 0.5 * (ts + ta)
    k, nu, al = props(tf)
    pr = nu / al
    beta = 1 / (tf + 273.15)
    dt = ts - ta
    if orient == "horizontal":
        lc = L / 2  # area/perimeter of a long strip, width L
        ra = 9.81 * beta * dt * lc ** 3 / (nu * al)
        return 0.5 * (0.54 + 0.27) * ra ** .25 * k / lc, ra
    g = 9.81 * np.cos(np.radians(orient))
    ra = g * beta * dt * L ** 3 / (nu * al)
    nu_l = 0.68 + 0.670 * ra ** .25 / (1 + (0.492 / pr) ** (9 / 16)) ** (4 / 9)
    return nu_l * k / L, ra


def flux(ts, ta, L, orient, eps):
    h, ra = h_conv(ts, ta, L, orient)
    return h * (ts - ta) + eps * SIGMA * ((ts + 273.15) ** 4 - (ta + 273.15) ** 4), h, ra


def analyse(L, orient, eps):
    l = [flux(SP, t, L, orient, eps)[0] for t in RT]
    off = [a - SP for a in A_MEAS]
    c = (off[1] - off[0]) / (l[1] - l[0])
    const = c * l[0] - off[0]
    f = lambda t: c * flux(SP, t, L, orient, eps)[0] - const
    rooms = np.linspace(15, 22, 29)
    o = np.array([f(t) for t in rooms])
    p = np.polyfit(rooms, o, 1)
    chord = np.interp(rooms, [15, 22], [o[0], o[-1]])
    # effective exponent of the loss law
    n = np.log(l[1] / l[0]) / np.log((SP - RT[1]) / (SP - RT[0]))
    return dict(o15=o[0], o20=f(20.0), dev_fit=np.abs(o - np.polyval(p, rooms)).max(),
                dev_chord=(o - chord).min(), slope22=(f(21.9) - f(22.1)) / -.2,
                slope15=(f(14.9) - f(15.1)) / -.2, n=n,
                kappa=const / (CHAMBER - SP), ra=flux(SP, 15.0, L, orient, eps)[2],
                h17=flux(SP, 17.5, L, orient, eps)[1], c=c, const=const)


rows = []
print(" L    angle  eps    n_eff  off@15  off@20  dev_line  slope22 slope15  Ra_max   h_conv")
for L in (.05, .1, .2, .3, .5):
    for orient in (0, 45, "horizontal"):
        for eps in (.04, .1, .72, .95):
            r = analyse(L, orient, eps)
            rows.append(r)
            if L in (.1, .3) and eps in (.1, .72):
                print(f"{L:4.2f} {str(orient):>6} {eps:4.2f} {r['n']:7.3f} {r['o15']:7.2f} "
                      f"{r['o20']:7.2f} {r['dev_fit']:8.3f} {r['slope22']:8.3f} "
                      f"{r['slope15']:7.3f} {r['ra']:8.1e} {r['h17']:6.2f}")

g = lambda k: np.array([r[k] for r in rows])
print("\nAcross all", len(rows), "combinations:")
for k, name in (("n", "effective exponent"), ("o15", "offset at 15"), ("o20", "offset at 20"),
                ("dev_fit", "deviation from straight line 15-22"),
                ("slope22", "slope at 22"), ("slope15", "slope at 15"),
                ("kappa", "kappa"), ("ra", "max Rayleigh")):
    print(f"  {name:38s} {g(k).min():.3g} to {g(k).max():.3g}")

print("\nReference cases against a purely linear law (exponent 1):")
line = lambda t: 1.0 + (5.25 - 1.0) * (22 - t) / 4.5
for t in (15, 16, 18, 20, 21):
    v = analyse(.3, 45, .1)
    print(f"  room {t}: straight line through measurements {line(t):.2f}")
print(f"  physical curve at 15: {analyse(.3, 45, .1)['o15']:.2f}, "
      f"at 20: {analyse(.3, 45, .1)['o20']:.2f}")

print("\nPower-law form for the nominal plate (L 0.3, 45 deg, eps 0.1):")
r = analyse(.3, 45, .1)
n = r["n"]
a = 4.25 / (17 ** n - 12.5 ** n)
b = a * 12.5 ** n - 1
print(f"  offset = {a:.4f} * (34.5 - room)^{n:.2f} - {b:.2f}")
L_, o_, e_ = .3, 45, .1
l = [flux(SP, t, L_, o_, e_)[0] for t in RT]
c = 4.25 / (l[1] - l[0])
const = c * l[0] - 1
err = max(abs(a * (34.5 - t) ** n - b - (c * flux(SP, t, L_, o_, e_)[0] - const))
          for t in np.arange(15, 23, .5))
print(f"  max error against the full physical law, 15-22.5: {err:.3f} K")
