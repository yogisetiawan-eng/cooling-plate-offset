"""Final QA: treat every reading as approximate and compare mechanisms.

Each draw perturbs the readings, solves the plate heat balance exactly, rejects
physically impossible cases and predicts the curves. The output is a median and
a 90% band.

Plate balance at true plate temperature Tp (the sensor reads a little low):
    A - Tp = c*L(Tp, room) - kappa*(Tres - Tp)
    Tp = B_read + phi*(B_read - room)        lead and air coupling at the sensor
    Tres = chamber + d                       heated floor beside the plate root

d = 0 is the 37 degC chamber path. Larger d is a hotter source, which behaves
more like a fixed heat input.
"""
import numpy as np

SIGMA = 5.67e-8
L_PLATE = 0.3
H = 1.42 / L_PLATE ** 0.25
TESTS = [(22.0, 35.5, 35.5, 35.5, 37.0), (17.5, 35.5, 39.8, 39.8, 37.0)]
ROOMS = np.array([15, 16, 17, 17.5, 18, 19, 20, 21, 22, 23], float)
N = 200_000
rng = np.random.default_rng(7)

# Reading errors (1 sigma), K. Pt100 class B at 35 degC is +/-0.475, taken as 2 sigma.
SD = dict(room_shared=.3, room_test=.5, b_shared=.25, b_test=.05,
          a_shared=.25, a_test=.15, air_shared=.2, air_test=.1)


def loss(t, room, n, eps):
    d = np.maximum(t - room, 1e-6)
    return H * d ** n + eps * SIGMA * ((t + 273.15) ** 4 - (room + 273.15) ** 4)


def draw(d_low=0.0, d_high=5.0, sd=SD):
    n = rng.uniform(1.15, 1.33, N)
    eps = rng.uniform(.05, .2, N)   # bare aluminium, no surface finish
    phi = rng.uniform(0, .03, N)
    d = rng.uniform(d_low, d_high, N)
    t1, t2 = TESTS
    def row(t):
        room = t[0] + rng.normal(0, sd["room_test"], N)
        b = t[1] + rng.normal(0, sd["b_test"], N)
        a = rng.uniform(t[2], t[3], N) + rng.normal(0, sd["a_test"], N)
        air = t[4] + rng.normal(0, sd["air_test"], N)
        return room, b, a, air
    r1, b1, a1, air1 = row(t1)
    r2, b2, a2, air2 = row(t2)
    rb, bb, ab, cb = (rng.normal(0, sd[k], N) for k in
                      ("room_shared", "b_shared", "a_shared", "air_shared"))
    r1, r2 = r1 + rb, r2 + rb
    b1, b2 = b1 + bb, b2 + bb
    a1, a2 = a1 + ab, a2 + ab
    air1, air2 = air1 + cb, air2 + cb
    tp1 = b1 + phi * (b1 - r1)
    tp2 = b2 + phi * (b2 - r2)
    o1, o2 = a1 - tp1, a2 - tp2
    l1, l2 = loss(tp1, r1, n, eps), loss(tp2, r2, n, eps)
    g1, g2 = air1 + d - tp1, air2 + d - tp2
    det = -l1 * g2 + l2 * g1
    c = (-o1 * g2 + o2 * g1) / det
    kappa = (l1 * o2 - l2 * o1) / det
    # A can sit below B now: the pad underside loses heat straight to the room.
    ok = (c > 0) & (kappa >= 0) & (g1 > 0) & (g2 > 0)
    return dict(n=n, eps=eps, phi=phi, d=d, c=c, kappa=kappa, ok=ok, bb=bb,
                a1=a1, o1=o1, o2=o2, l1=l1, l2=l2, cb=cb, rb=rb, ab=ab)


def offset(p, b, room, air=37.0):
    """A reading minus B reading, given sensor READINGS of B, room and chamber.
    Each sensor keeps the same bias it had during the tests, so biases cancel."""
    bt, rt, at = b + p["bb"], room + p["rb"], air + p["cb"]
    tp = bt + p["phi"] * (bt - rt)
    a_true = tp + p["c"] * loss(tp, rt, p["n"], p["eps"]) - p["kappa"] * (at + p["d"] - tp)
    return a_true - p["ab"] - b


def setpoint(p, target, room):
    lo = np.full_like(p["c"], 20.0)
    hi = np.full_like(p["c"], 45.0)
    for _ in range(40):
        m = .5 * (lo + hi)
        hot = m + offset(p, m, room) > target
        hi = np.where(hot, m, hi)
        lo = np.where(hot, lo, m)
    return .5 * (lo + hi)


def subset(p):
    k = p["ok"]
    return {x: (v[k] if isinstance(v, np.ndarray) else v) for x, v in p.items()}


def band(x):
    return np.percentile(x, [5, 50, 95])


p_raw = draw()
print(f"Draws kept after physical limits: {p_raw['ok'].mean() * 100:.1f}%")
p = subset(p_raw)

print("\n1. Could reading errors alone make a simple series path work?")
l_ratio = p_raw["l2"] / p_raw["l1"]
expected = p_raw["o1"] * l_ratio
print(f"   Offset at 17.5 expected from series path: {np.median(expected):.2f} K")
print(f"   Measured, with errors: {band(p_raw['o2']).round(2)}")
print(f"   Chance readings alone close the gap: {np.mean(expected >= p_raw['o2']) * 100:.3f}%")
print("   Error at both sensors needed: "
      f"{(4.3 - 0.0) / np.hypot(np.hypot(SD['a_test'], .14), np.hypot(SD['b_test'], SD['room_test'] * .94)):.1f} sigma")

print("\n2. Offset at fixed B reading 35.5")
print("room   5%   median  95%   width")
rows = []
for t in ROOMS:
    o = offset(p, 35.5, t)
    lo, med, hi = band(o)
    rows.append(med)
    print(f"{t:5.1f} {lo:5.2f} {med:6.2f} {hi:5.2f}  {hi - lo:4.2f}")
s = np.polyfit(ROOMS[:9], rows[:9], 1)
print(f"Median line, 15-22: offset = {s[1]:.2f} {s[0]:+.3f} * room, "
      f"max error {np.abs(np.polyval(s, ROOMS[:9]) - np.array(rows[:9])).max():.3f} K")
sl = (offset(p, 35.5, 15.) - offset(p, 35.5, 22.)) / (15 - 22)
print(f"Slope 15-22, K/K: {band(sl).round(3)}")

print("\n3. B setpoint to hold A at 35.5, and the implied offset")
print("room  5%   median  95%   offset median")
for t in ROOMS:
    sp = setpoint(p, 35.5, t)
    lo, med, hi = band(sp)
    print(f"{t:5.1f} {lo:5.2f} {med:6.2f} {hi:5.2f}  {35.5 - med:5.2f}")

print("\n4. How the mechanism changes the setpoint at 17.5 degC room")
print("d  gain 22C  setpoint  A if 34.59 is used")
for dlo, dhi in ((0, 0), (1, 1), (2, 2), (3, 3), (5, 5), (10, 10), (1e3, 1e3)):
    q = subset(draw(dlo, dhi))
    e = 1e-3
    g = 1 + (offset(q, 35.5 + e, 22.) - offset(q, 35.5 - e, 22.)) / (2 * e)
    sp = setpoint(q, 35.5, 17.5)
    a = 34.59 + offset(q, 34.59, 17.5)
    print(f"{dlo:6.0f} {np.median(g):5.2f} {np.median(sp):8.2f} {np.median(a):8.2f}")

print("\n5. Gain over the prior")
e = 1e-3
g = 1 + (offset(p, 35.5 + e, 22.) - offset(p, 35.5 - e, 22.)) / (2 * e)
print("   gain 5/50/95:", band(g).round(2))

print("\n6. Heater power, pad open to the room underneath")
AREA, PMAX = .068434, 30.0
SIDE = AREA ** .5
print(f"   {AREA*1e6:.0f} mm2 = {SIDE*1e3:.0f} mm square, {PMAX/AREA:.0f} W/m2, {48**2/PMAX:.1f} ohm, {PMAX/48:.3f} A")
m_h = rng.uniform(.6, 1.4, N)      # pad underside between the plate legs
e_pad = rng.uniform(.8, .95, N)
h_in = rng.uniform(2, 8, N)        # chamber side, W/m2K
g_ab = np.exp(rng.uniform(np.log(.2), np.log(1), N))
fed = rng.uniform(0, 1, N)         # share of the bypass heat that comes from the pad


def power(a, b, room, air=37.0):
    d = np.maximum(a - room, 1e-3)
    h_u = m_h * .59 * (d / SIDE) ** .25
    rad = e_pad * SIGMA * ((a + 273.15) ** 4 - (room + 273.15) ** 4)
    return AREA * (h_u * d + rad + h_in * (a - air)) + g_ab * (a - b) + fed * g_ab * 11.0


q22 = power(35.5, 35.5, 22.)
q17 = power(39.8, 35.5, 17.5)
print("   tested, 22 degC room: W", band(q22).round(1), "duty %", band(q22 / PMAX * 100).round(0))
print("   tested, 17.5 degC room: W", band(q17).round(1), "duty %", band(q17 / PMAX * 100).round(0))
print("   power ratio 17.5 vs 22:", band(q17 / q22).round(2))
lim = (PMAX - AREA * (m_h * .59 * (22.3 / SIDE) ** .25 * 22.3
       + e_pad * SIGMA * (312.95 ** 4 - 290.65 ** 4) + h_in * 2.8)) / 4.3
print("   largest A-to-plate conductance 30 W allows, W/K:", band(lim).round(1), f"absolute ceiling {PMAX / 4.3:.1f}")
print(f"   same link as areal R, m2K/W, at G 1 W/K: {AREA / 1:.3f}; silicone 0.8-1.5 mm: 0.0033-0.0067")
for t, b_ in ((15., 34.09), (22., 35.5), (28., 36.64), (30., 36.99), (33., 37.6)):
    print(f"   hold A 35.5, room {t}: W", band(power(35.5, b_, t)).round(1))

print("\n7. What a step test would buy")
true_gains = []
sps = []
for d_true in (0., 2., 5.):
    t = subset(draw(d_true, d_true))
    g_ = 1 + (offset(t, 35.5 + e, 22.) - offset(t, 35.5 - e, 22.)) / (2 * e)
    sp = setpoint(t, 35.5, 17.5)
    print(f"   d={d_true}: gain {np.median(g_):.2f}, A rise for +0.5 K at B {0.5 * np.median(g_):.2f} K, "
          f"setpoint at 17.5 {band(sp).round(2)}")

print("\n8. Sensible size checks")
area = .3 * .2 * 2
for h_tot in (4.0, 8.0):
    gbr = h_tot * area
    print(f"   Plate 0.3 x 0.2 m both faces, h {h_tot}: room conductance {gbr:.2f} W/K, "
          f"loss at 13.5 K {gbr * 13.5:.1f} W")
    print(f"     Implied G_AB about {gbr / .94:.2f} W/K")
r_pad = (.0033, .0067)
a_pad = .06
print(f"   A silicone layer 0.8-1.5 mm over {a_pad} m2 would give "
      f"{a_pad / r_pad[1]:.0f} to {a_pad / r_pad[0]:.0f} W/K")


def psat(t):
    return 0.61094 * np.exp(17.625 * t / (243.04 + t))


print("\n9. If A sets humidity by being the coldest wet spot")
for a in (35, 35.5, 36, 36.5, 39.8):
    print(f"   A {a}: RH in 37 degC air {psat(a) / psat(37) * 100:.1f}%")
print(f"   Check: dew point of 37 degC, 98% RH = "
      f"{243.04 * np.log(.98 * psat(37) / .61094) / (17.625 - np.log(.98 * psat(37) / .61094)):.1f}")
