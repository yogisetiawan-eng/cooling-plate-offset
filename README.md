# Cooling Plate Offset

A calculator for a CO2 incubator cooling plate. It gives the B sensor setpoint that holds the inner chamber floor at a target temperature in any room, along with the offset between the two.

Open `index.html` in a browser. It is one self-contained page and needs no install or build.

## Layout

- **A:** inner bottom surface of the chamber, over the heater pad. Used for logging only.
- **B:** control sensor on the aluminium plate, outside the insulation.
- **Heater pad:** silicone, 68,434 mm², 30 W at 48 V. There is no insulation under it, so it loses heat directly to the room.
- **Plate:** bare aluminium, running from the pad edges through the insulation to room air.
- **Chamber wall:** 316 stainless, 1 mm.

## How to use it

1. Enter the room temperature, chamber air and A target.
2. Read the B setpoint, the offset and the estimated heater power.
3. Add each new steady-state test to the Test data table. The model refits itself. Rows are saved in that browser only.

## Current result

For an A target of 35.5 °C and chamber air of 37 °C:

```
B setpoint = 31.22 + 0.193 × room
Offset, A − B = 4.28 − 0.193 × room
```

This comes from two tests with B at 35.5 °C: A read 35.5 °C at 22 °C room and 39.8 °C at 17.5 °C room. It is not validated yet. How strongly B moves A is still uncertain, at about 3.3 to 10 K per 1 K. A test at a new B setpoint narrows that range. The page suggests the next test.

## Method

The plate heat balance at B is:

```
A − Tp = c × L(Tp, room) − κ × (Tsource − Tp) − K0
```

- `L` is plate heat loss to the room, from laminar natural convection plus radiation.
- `Tsource` is chamber air, up to 5 K hotter, because the floor beside the plate may be heated.
- `Tp` is the true plate temperature. The sensor can read slightly low.

The page runs 400 repeatable draws. Each one varies the sensor readings, loss law and source temperature within their expected ranges, then solves the balance. The result is the median with a 90% range.

## Assumptions to replace with real data

- Plate length L of 0.3 m.
- Reading errors of about ±1 K for room, ±0.5 K for A and B, and ±0.4 K for chamber air.
- Heater power is an estimate. It assumes the pad underside is open to room air.

## Analysis scripts

- `analysis/qa_study.py`: the uncertainty and mechanism study behind the page.
- `analysis/linearity_study.py`: how linear the offset is across plate lengths, angles and surface finishes.

Both need Python with NumPy.

## References

- Churchill and Chu, Int. J. Heat Mass Transfer 18, 1323 (1975): natural convection on a vertical plate.
- Simons, Electronics Cooling, August 2001: simplified flat plate natural convection.
- Cleveland-Cliffs 316/316L data sheet: 16.2 W/m·K at 100 °C.
