import type { ReactNode } from "react";
import { bernstein, classFunction, leadingEdgeMode } from "./kulfan";
import { Chart, Sheet, U, linear, pathFrom, type Point } from "./plot";

// One small square is 0.02 chord across and 0.005 chord up.
const W = 56 * U;
const H = 39 * U;
const box = { left: 4 * U, right: 54 * U, top: 2 * U, bottom: 34 * U };
const x = linear(0, 1, box.left, box.right);
const y = linear(0, 0.16, box.bottom, box.top);

const samples = Array.from({ length: 201 }, (_, i) => i / 200);
const mode = (k: number): Point[] =>
    samples.map((s) => [x(s), y(classFunction(s) * bernstein(k, s))]);
const peak = (k: number): Point =>
    samples
        .map((s): Point => [s, classFunction(s) * bernstein(k, s)])
        .reduce((best, point) => (point[1] > best[1] ? point : best));

/** The eight shape modes of one surface and the leading-edge mode. */
export default function KulfanModes(): ReactNode {
    return (
        <figure className="bora-figure">
            <div className="bora-plate">
                <svg
                    className="bora-plot"
                    viewBox={`0 0 ${W} ${H}`}
                    role="img"
                    aria-label="Chart of the eight Kulfan shape modes against chordwise position, numbered 0 at the leading edge to 7 at the trailing edge, with the leading-edge mode dashed."
                >
                    <Sheet width={W} height={H} />
                    <Chart
                        box={box}
                        x={x}
                        y={y}
                        xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                        yTicks={[0, 0.05, 0.1, 0.15]}
                        xFormat={(v) => (v === 0 ? "0" : v.toFixed(1))}
                        yFormat={(v) => (v === 0 ? "0" : v.toFixed(2))}
                        xTitle="x/c"
                    />
                    <text x={box.right - 8} y={box.top + 20} textAnchor="end">
                        y/c for unit weight
                    </text>
                    {Array.from({ length: 8 }, (_, k) => (
                        <path key={k} className="line" d={pathFrom(mode(k))} />
                    ))}
                    <path
                        className="line line--dashed"
                        d={pathFrom(samples.map((s) => [x(s), y(leadingEdgeMode(s))]))}
                    />
                    {Array.from({ length: 8 }, (_, k) => {
                        const [px, py] = peak(k);
                        return (
                            <text key={k} x={x(px)} y={y(py) - 7} textAnchor="middle">
                                {k}
                            </text>
                        );
                    })}
                    <path
                        className="tick"
                        d={`M${x(0.16)} ${y(0.034)} L${x(0.3)} ${y(0.012)} h14`}
                    />
                    <text x={x(0.3) + 20} y={y(0.012) + 5}>
                        leading-edge mode
                    </text>
                </svg>
            </div>
            <figcaption>
                The shape modes of one surface. Each solid curve is the surface that results from
                setting a single weight to one, numbered by weight; the dashed curve is the
                leading-edge mode. An airfoil surface is a weighted sum of these nine curves plus a
                linear trailing-edge thickness term.
            </figcaption>
        </figure>
    );
}
