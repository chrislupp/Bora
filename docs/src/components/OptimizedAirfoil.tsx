import type { ReactNode } from "react";
import { stations, surface, type Kulfan } from "./kulfan";
import { Sheet, U, linear, pathFrom, type Point } from "./plot";

// Start and end of examples/optimize_airfoil.rs.
const start: Kulfan = {
    upper: [0.172884, 0.151563, 0.173763, 0.127681, 0.164818, 0.126376, 0.145898, 0.1392],
    lower: [-0.172884, -0.151563, -0.173763, -0.127681, -0.164818, -0.126376, -0.145898, -0.1392],
    leadingEdge: 0,
    teThickness: 0.002548,
};
const optimized: Kulfan = {
    upper: [0.097579, 0.146209, 0.21594, 0.161252, 0.254478, 0.300737, 0.213287, 0.125876],
    lower: [-0.058255, 0.027221, -0.067101, -0.050751, -0.087094, -0.033425, -0.016107, -0.03255],
    leadingEdge: 0.191559,
    teThickness: 0.002548,
};

// One heavy ruling is a tenth of the chord.
const W = 56 * U;
const H = 17 * U;
const CHORD = 50 * U;
const X0 = 3 * U;
const YC = 8 * U;
const x = linear(0, 1, X0, X0 + CHORD);
const y = linear(0, 1, YC, YC - CHORD);
const xs = stations(81);

const outline = (shape: Kulfan): string => {
    const upper: Point[] = xs.map((s): Point => [x(s), y(surface(shape, "upper", s))]).reverse();
    const lower: Point[] = xs.map((s): Point => [x(s), y(surface(shape, "lower", s))]);
    return `${pathFrom([...upper, ...lower])} Z`;
};

/** The starting and optimized sections of the optimization example. */
export default function OptimizedAirfoil(): ReactNode {
    const startY = y(surface(start, "lower", 0.3));
    const endY = y(surface(optimized, "upper", 0.62));
    return (
        <figure className="bora-figure">
            <div className="bora-plate">
                <svg
                    className="bora-plot"
                    viewBox={`0 0 ${W} ${H}`}
                    role="img"
                    aria-label="Outlines of the starting NACA 0012 section, dashed, and the optimized section, solid. The optimized section is thinner, with camber toward the rear."
                >
                    <Sheet width={W} height={H} />
                    <path className="line line--dashed" d={outline(start)} />
                    <path className="line line--heavy" d={outline(optimized)} />
                    {[0, 0.5, 1].map((s) => (
                        <text key={s} x={x(s)} y={H - 10} textAnchor="middle">
                            {s === 0 ? "0" : s.toFixed(1)}
                        </text>
                    ))}
                    <text x={x(0.25)} y={H - 10} textAnchor="middle">
                        x/c
                    </text>
                    <path className="tick" d={`M${x(0.3)} ${startY + 3} l20 16 h12`} />
                    <text x={x(0.3) + 38} y={startY + 24}>
                        start, NACA 0012
                    </text>
                    <path className="tick" d={`M${x(0.62)} ${endY - 3} l20 -16 h12`} />
                    <text x={x(0.62) + 38} y={endY - 14}>
                        optimized
                    </text>
                </svg>
            </div>
            <figcaption>
                The starting section and the section after 500 iterations, drawn from the Kulfan
                parameters printed above. One heavy ruling is a tenth of the chord.
            </figcaption>
        </figure>
    );
}
