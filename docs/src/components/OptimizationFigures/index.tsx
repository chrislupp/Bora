import type { ReactNode } from "react";
import data from "@site/src/data/optimization.json";
import { stations, surface, type Kulfan } from "../kulfan";
import {
    Chart,
    PlotSymbol,
    Sheet,
    U,
    Y_TITLE_ROOM,
    linear,
    minus,
    pathFrom,
    type Point,
} from "../plot";
import styles from "./styles.module.css";

/*
 * Figures of the OpenMDAO optimization example. Every number is output of
 * examples/openmdao/optimize_daedalus.py; regenerate the data with
 *
 *   python3 examples/openmdao/optimize_daedalus.py --json docs/src/data/optimization.json
 *
 * The NACA 0012 start is dashed and the optimized section solid throughout;
 * the six design points are open circles.
 */

type Design = typeof data.optimized;
type Row = Design["schedule"][number];

const kulfanOf = (design: Design): Kulfan => ({
    upper: design.kulfan.upper,
    lower: design.kulfan.lower,
    leadingEdge: design.kulfan.leading_edge,
    teThickness: design.kulfan.te_thickness,
});

const start = kulfanOf(data.start);
const optimized = kulfanOf(data.optimized);
const limits = data.constraints;

const fixed = (digits: number) => (v: number) => (v === 0 ? "0" : v.toFixed(digits));

/* ── Sections ─────────────────────────────────────────────────────────── */

// One small square is 0.02 chord; one heavy ruling is a tenth of the chord.
const SECTION_W = 56 * U;
const SECTION_H = 19 * U;
const CHORD = 50 * U;
const sx = linear(0, 1, 3 * U, 3 * U + CHORD);
const sy = linear(0, 1, 10 * U, 10 * U - CHORD);
const xs = stations(121);

const outline = (shape: Kulfan): string => {
    const upper: Point[] = xs.map((s): Point => [sx(s), sy(surface(shape, "upper", s))]).reverse();
    const lower: Point[] = xs.map((s): Point => [sx(s), sy(surface(shape, "lower", s))]);
    return `${pathFrom([...upper, ...lower])} Z`;
};

/** A thickness requirement drawn as a dimension line between the surfaces. */
function Spar({ station, minimum }: { station: number; minimum: number }): ReactNode {
    const top = surface(optimized, "upper", station);
    const bottom = surface(optimized, "lower", station);
    const middle = (top + bottom) / 2;
    const x = sx(station);
    const y0 = sy(middle + minimum / 2);
    const y1 = sy(middle - minimum / 2);
    return (
        <g>
            <path className="mark mark--thin" d={`M${x} ${y0} V${y1}`} />
            <path className="mark mark--thin" d={`M${x - 5} ${y0} h10 M${x - 5} ${y1} h10`} />
        </g>
    );
}

/** The starting and optimized sections, with the two spar requirements. */
export function OptimizedSections(): ReactNode {
    const [main, rear] = limits.spar_stations;
    const [mainMin, rearMin] = limits.spar_thickness_min;
    const mainTop = sy(surface(optimized, "upper", main));
    const rearBottom = sy(surface(optimized, "lower", rear));
    const startLabel = sy(surface(start, "lower", 0.2));
    const optimizedLabel = sy(surface(optimized, "upper", 0.62));
    return (
        <figure className="bora-figure">
            <div className="bora-panel">
                <svg
                    className="bora-plot"
                    viewBox={`0 0 ${SECTION_W} ${SECTION_H}`}
                    role="img"
                    aria-label={`Outlines of the starting NACA 0012 section, dashed, and the optimized section, solid. The optimized section has a nearly flat lower surface and a thick, strongly cambered upper surface. Dimension lines at ${main} and ${rear} chord show the required thicknesses of ${mainMin} and ${rearMin} chord, which the optimized section meets exactly.`}
                >
                    <Sheet width={SECTION_W} height={SECTION_H} />
                    <path className="line line--dashed" d={outline(start)} />
                    <path className="line line--heavy" d={outline(optimized)} />
                    <Spar station={main} minimum={mainMin} />
                    <Spar station={rear} minimum={rearMin} />

                    <path className="tick" d={`M${sx(main)} ${mainTop - 6} v-${1.5 * U} h-12`} />
                    <text x={sx(main) - 18} y={mainTop - 1.5 * U + 5} textAnchor="end">
                        main spar, t/c ≥ {mainMin}
                    </text>
                    <path
                        className="tick"
                        d={`M${sx(rear)} ${rearBottom + 6} V${sy(-0.075)} h-12`}
                    />
                    <text x={sx(rear) - 18} y={sy(-0.075) + 5} textAnchor="end">
                        rear spar, t/c ≥ {rearMin}
                    </text>

                    <path className="tick" d={`M${sx(0.2)} ${startLabel + 4} l16 16 h12`} />
                    <text x={sx(0.2) + 34} y={startLabel + 25}>
                        start, NACA 0012
                    </text>
                    <path className="tick" d={`M${sx(0.62)} ${optimizedLabel - 4} l16 -16 h12`} />
                    <text x={sx(0.62) + 34} y={optimizedLabel - 15}>
                        optimized
                    </text>

                    {[0, 0.5, 1].map((s) => (
                        <text key={s} x={sx(s)} y={SECTION_H - 10} textAnchor="middle">
                            {fixed(1)(s)}
                        </text>
                    ))}
                    <text x={sx(0.25)} y={SECTION_H - 10} textAnchor="middle">
                        x/c
                    </text>
                </svg>
            </div>
            <figcaption>
                The starting NACA 0012 and the optimized section. The dimension lines at the two
                spar stations have the required thicknesses; at the optimum both meet the surfaces
                exactly, so both constraints are active. One small square is 0.02 of the chord.
            </figcaption>
        </figure>
    );
}

/* ── Along the flight schedule ────────────────────────────────────────── */

// Each chart spans the column. Lift is up the sheet at 0.05 per small square,
// so the heavy rulings fall on every 0.25; drag is 0.001 and moment 0.005 per
// small square across.
const SCHEDULE_W = 62 * U;
const SCHEDULE_H = 37 * U;
const LIFT_MIN = 0.4;
const scheduleBox = { left: 10 * U, right: 60 * U, top: 4 * U, bottom: 32 * U };
const ly = linear(0.5, 1.0, 30 * U, 20 * U);
const CD_MAX = 0.05;

const trusted = (rows: Row[]): Row[] =>
    rows.filter((r) => r.confidence >= limits.confidence_min && r.cl >= LIFT_MIN);
// Stop each curve at its maximum lift: past it the polar folds back.
const toStall = (rows: Row[]): Row[] => {
    const peak = rows.reduce((best, r, i) => (r.cl > rows[best].cl ? i : best), 0);
    return rows.slice(0, peak + 1);
};
const startRows = toStall(trusted(data.start.schedule));
const optimizedRows = toStall(trusted(data.optimized.schedule));
const points = data.optimized.points;

function DragPolar(): ReactNode {
    const box = scheduleBox;
    const x = linear(0, CD_MAX, box.left, box.right);
    const inside = (r: Row): boolean => r.cd <= CD_MAX;
    const startEnd = startRows.filter(inside).at(-1)!;
    return (
        <svg
            className="bora-plot"
            viewBox={`0 0 ${SCHEDULE_W} ${SCHEDULE_H}`}
            role="img"
            aria-label={`Drag polars along the flight schedule. The NACA 0012 reaches a lift coefficient of only ${startRows.at(-1)!.cl.toFixed(2)}; the optimized section reaches ${optimizedRows.at(-1)!.cl.toFixed(2)} and passes through all six design points with drag coefficients from ${points[0].cd.toFixed(4)} to ${points.at(-1)!.cd.toFixed(4)}.`}
        >
            <Sheet width={SCHEDULE_W} height={SCHEDULE_H} />
            <Chart
                box={box}
                x={x}
                y={ly}
                xTicks={[0, 0.01, 0.02, 0.03, 0.04, 0.05]}
                yTicks={[0.5, 1, 1.5]}
                xFormat={fixed(2)}
                yFormat={fixed(1)}
                xTitle="Section drag coefficient"
                yTitle="Section lift coefficient"
            />
            <path
                className="line line--dashed"
                d={pathFrom(startRows.filter(inside).map((r) => [x(r.cd), ly(r.cl)]))}
            />
            <path
                className="line line--heavy"
                d={pathFrom(optimizedRows.filter(inside).map((r) => [x(r.cd), ly(r.cl)]))}
            />
            {points.map((p) => (
                <PlotSymbol key={p.cl_target} kind="circle" x={x(p.cd)} y={ly(p.cl)} />
            ))}
            <text x={x(startEnd.cd) - 8} y={ly(startEnd.cl) + 22} textAnchor="end">
                NACA 0012
            </text>
            <text x={x(0.0135)} y={ly(1.45) + 5}>
                optimized
            </text>
        </svg>
    );
}

function MomentChart(): ReactNode {
    const box = scheduleBox;
    const x = linear(-0.2, 0.05, box.left, box.right);
    const limitX = x(limits.cm_min);
    return (
        <svg
            className="bora-plot"
            viewBox={`0 0 ${SCHEDULE_W} ${SCHEDULE_H}`}
            role="img"
            aria-label={`Pitching moment against lift along the flight schedule. The NACA 0012 has a moment coefficient near zero. The optimized section has a moment between ${minus(Math.max(...optimizedRows.map((r) => r.cm)).toFixed(3))} and ${minus(Math.min(...optimizedRows.map((r) => r.cm)).toFixed(3))}, touching the limit of ${minus(limits.cm_min.toString())} near a lift coefficient of 1.5.`}
        >
            <Sheet width={SCHEDULE_W} height={SCHEDULE_H} />
            <Chart
                box={box}
                x={x}
                y={ly}
                xTicks={[-0.2, -0.15, -0.1, -0.05, 0, 0.05]}
                yTicks={[0.5, 1, 1.5]}
                xFormat={fixed(2)}
                yFormat={fixed(1)}
                xTitle="Section moment coefficient"
                yTitle="Section lift coefficient"
            />
            <path className="mark mark--thin" d={`M${limitX} ${box.top} V${box.bottom}`} />
            <text x={limitX - 8} y={ly(1.05) + 5} textAnchor="end">
                {minus(`limit ${limits.cm_min}`)}
            </text>
            <path
                className="line line--dashed"
                d={pathFrom(startRows.map((r) => [x(r.cm), ly(r.cl)]))}
            />
            <path
                className="line line--heavy"
                d={pathFrom(optimizedRows.map((r) => [x(r.cm), ly(r.cl)]))}
            />
            {points.map((p) => (
                <PlotSymbol key={p.cl_target} kind="circle" x={x(p.cm)} y={ly(p.cl)} />
            ))}
            <text x={x(0) + 12} y={ly(0.7) + 5}>
                NACA 0012
            </text>
            <text x={x(-0.115)} y={ly(1.3) + 5}>
                optimized
            </text>
        </svg>
    );
}

/** Drag polar and moment of both sections along the flight schedule. */
export function SchedulePolars(): ReactNode {
    return (
        <figure className="bora-figure">
            <div className={`bora-panel ${styles.stack}`}>
                <DragPolar />
                <MomentChart />
            </div>
            <figcaption>
                Both sections flown along the schedule of the problem, each angle of attack
                evaluated at the Reynolds number its own lift implies. The circles are the six
                design points of the optimum; they lie on its curves. Each curve stops at maximum
                lift and where the analysis confidence falls below {limits.confidence_min}. The
                vertical line is the moment limit. One small square is 0.05 in lift, 0.001 in drag
                and 0.005 in moment.
            </figcaption>
        </figure>
    );
}

/* ── Convergence ──────────────────────────────────────────────────────── */

// Two iterations per small square across and one decade per small square up,
// with the powers of ten on heavy rulings. Optimality is IPOPT's dual
// infeasibility (inf_du), feasibility its primal infeasibility (inf_pr).
const HISTORY_W = 62 * U;
const HISTORY_H = 42 * U;
const history = data.history;
const hx = linear(0, 100, 10 * U, 60 * U);
const ITERATION_TICKS = [0, 20, 40, 60, 80, 100];

/** A log chart: its box, the decade at the box top, the labelled decades and the title. */
type LogScale = {
    box: { left: number; right: number; top: number; bottom: number };
    top: number;
    ticks: number[];
    title: string;
};

const optimalityScale: LogScale = {
    box: { left: 10 * U, right: 60 * U, top: 3 * U, bottom: 15 * U },
    top: 2,
    ticks: [0, -5, -10],
    title: "Optimality",
};
const feasibilityScale: LogScale = {
    box: { left: 10 * U, right: 60 * U, top: 20 * U, bottom: 36 * U },
    top: 0,
    ticks: [0, -5, -10, -15],
    title: "Feasibility",
};

/** Height on a log scale; values below its floor sit on the floor. */
const logY = (scale: LogScale, value: number): number => {
    const floor = scale.top - (scale.box.bottom - scale.box.top) / U;
    const decade = Math.log10(Math.max(value, 10 ** floor));
    return scale.box.top + (scale.top - decade) * U;
};

function Power({ exponent }: { exponent: number }): ReactNode {
    if (exponent === 0) {
        return <>1</>;
    }
    return (
        <>
            10
            <tspan dy={-6} fontSize="10px">
                {minus(String(exponent))}
            </tspan>
        </>
    );
}

/** Left and bottom axes of a log chart; only the lowest chart numbers the iterations. */
function LogAxes({ scale, iterations }: { scale: LogScale; iterations: boolean }): ReactNode {
    const { box } = scale;
    // Powers of ten are wider than plain numerals, so the title stands further out.
    const titleX = box.left - Y_TITLE_ROOM - 1.5 * U;
    const titleY = (box.top + box.bottom) / 2;
    return (
        <g>
            <path className="frame" d={`M${box.left} ${box.top} V${box.bottom} H${box.right}`} />
            {ITERATION_TICKS.map((t) => (
                <g key={t}>
                    <path className="tick" d={`M${hx(t)} ${box.bottom} v6`} />
                    {iterations && (
                        <text x={hx(t)} y={box.bottom + 2 * U - 2} textAnchor="middle">
                            {t}
                        </text>
                    )}
                </g>
            ))}
            {scale.ticks.map((e) => (
                <g key={e}>
                    <path className="tick" d={`M${box.left} ${logY(scale, 10 ** e)} h-6`} />
                    <text x={box.left - 10} y={logY(scale, 10 ** e) + 5} textAnchor="end">
                        <Power exponent={e} />
                    </text>
                </g>
            ))}
            {iterations && (
                <text x={(box.left + box.right) / 2} y={box.bottom + 4 * U - 2} textAnchor="middle">
                    IPOPT iteration
                </text>
            )}
            <text
                x={titleX}
                y={titleY}
                textAnchor="middle"
                transform={`rotate(-90 ${titleX} ${titleY})`}
            >
                {scale.title}
            </text>
        </g>
    );
}

/** Optimality and feasibility at each IPOPT iteration. */
export function Convergence(): ReactNode {
    const first = history[0];
    const last = history.at(-1)!;
    return (
        <figure className="bora-figure">
            <div className="bora-panel">
                <svg
                    className="bora-plot"
                    viewBox={`0 0 ${HISTORY_W} ${HISTORY_H}`}
                    role="img"
                    aria-label={`IPOPT convergence history over ${last.iter} iterations. Optimality, the dual infeasibility, falls from ${first.optimality} to ${last.optimality.toExponential(1)}; feasibility, the largest constraint violation, falls from ${first.feasibility} to ${last.feasibility.toExponential(1)}, with spikes where the optimizer takes larger steps.`}
                >
                    <Sheet width={HISTORY_W} height={HISTORY_H} />
                    <LogAxes scale={optimalityScale} iterations={false} />
                    <path
                        className="line line--heavy"
                        d={pathFrom(
                            history.map((h) => [hx(h.iter), logY(optimalityScale, h.optimality)]),
                        )}
                    />
                    <LogAxes scale={feasibilityScale} iterations />
                    <path
                        className="line line--heavy"
                        d={pathFrom(
                            history.map((h) => [hx(h.iter), logY(feasibilityScale, h.feasibility)]),
                        )}
                    />
                </svg>
            </div>
            <figcaption>
                Optimality (IPOPT&apos;s dual infeasibility) and feasibility (its largest constraint
                violation) at each of the {last.iter} IPOPT iterations, read from IPOPT&apos;s
                output file. One small square is two iterations across and one decade up.
            </figcaption>
        </figure>
    );
}
