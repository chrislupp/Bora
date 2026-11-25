import type { ReactNode } from "react";
import data from "@site/src/data/polar.json";
import { Chart, PlotSymbol, SYMBOLS, Sheet, U, linear, minus, pathFrom, type Point } from "../plot";
import styles from "./styles.module.css";

/*
 * The section characteristics of the quickstart airfoil at three Reynolds
 * numbers, as a static figure. Every number shown is Bora output computed
 * ahead of time; regenerate the data with
 *
 *   cargo run --release --example polar -- --json > docs/src/data/polar.json
 *
 * The charts are scaled to the sheet: one small square is 1 degree across,
 * 0.1 in lift coefficient up, and 0.002 in drag coefficient across. One
 * condition is marked: the quickstart's own, 3 degrees at a Reynolds number
 * of one million.
 */

type Row = (typeof data.series)[number]["polar"][number];

const series = data.series;
const outline = data.outline as Point[];

const MARKED_SERIES = 1; // Reynolds number 1e6
const MARKED_ALPHA = 3;
const rows = series[MARKED_SERIES].polar;
const markedIndex = rows.findIndex((p) => p.alpha === MARKED_ALPHA);
const marked = rows[markedIndex];

// Selig ordering: upper surface from trailing to leading edge, then the lower
// surface back. Both are kept here from the leading edge aft.
const nose = outline.reduce((best, point, i) => (point[0] < outline[best][0] ? i : best), 0);
const upper = outline.slice(0, nose + 1).reverse();
const lower = outline.slice(nose);

const ALPHA: Point = [series[0].polar[0].alpha, series[0].polar.at(-1)!.alpha];
const LIFT: Point = [-0.5, 2.0];
const CD_MAX = 0.04;
const LIFT_COLS = ALPHA[1] - ALPHA[0]; // 26 squares
const DRAG_COLS = CD_MAX / 0.002; // 20 squares
const ROWS = (LIFT[1] - LIFT[0]) / 0.1; // 25 squares
const SYMBOL_STEP = 8; // one symbol every 2 degrees

const millions = (re: number): string => (re / 1e6).toFixed(1);

/** Height of a surface (leading edge aft) at chordwise station `x`. */
function heightAt(surface: Point[], x: number): number {
    const i = Math.max(
        1,
        surface.findIndex((point) => point[0] >= x),
    );
    const [x0, y0] = surface[i - 1];
    const [x1, y1] = surface[i];
    return x1 === x0 ? y0 : y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
}

/* ── The three drawings, each placed at an origin on the sheet ── */

function Profile({ x0, yc, chord }: { x0: number; yc: number; chord: number }): ReactNode {
    const at = (p: Point): Point => [x0 + p[0] * chord, yc - p[1] * chord];
    const [topX, topY] = at([marked.top_xtr, heightAt(upper, marked.top_xtr)]);
    const [bottomX, bottomY] = at([marked.bot_xtr, heightAt(lower, marked.bot_xtr)]);
    return (
        <g>
            <path className="paper" d={`${pathFrom(outline.map(at))} Z`} />
            <path className="line line--heavy" d={`${pathFrom(outline.map(at))} Z`} />
            <path className="tick" d={`M${topX.toFixed(1)} ${topY - 5} v-18`} />
            <text x={topX} y={topY - 29} textAnchor="middle">
                {marked.top_xtr.toFixed(2)}
            </text>
            <path className="tick" d={`M${bottomX.toFixed(1)} ${bottomY + 5} v18`} />
            <text x={bottomX} y={bottomY + 38} textAnchor="middle">
                {marked.bot_xtr.toFixed(2)}
            </text>
        </g>
    );
}

function LiftChart({ x0, y0 }: { x0: number; y0: number }): ReactNode {
    const box = { left: x0, right: x0 + LIFT_COLS * U, top: y0, bottom: y0 + ROWS * U };
    const x = linear(ALPHA[0], ALPHA[1], box.left, box.right);
    const y = linear(LIFT[0], LIFT[1], box.bottom, box.top);
    // Tangent to the lift curve from the analytic derivative.
    const span = 4.5;
    const tangent: Point[] = [
        [x(marked.alpha - span), y(marked.cl - span * marked.dcl_dalpha)],
        [x(marked.alpha + span), y(marked.cl + span * marked.dcl_dalpha)],
    ];
    const mark: Point = [x(marked.alpha), y(marked.cl)];
    // The slope is lettered in the empty corner below the curves, on a leader
    // that drops from the mark and runs to the lettering.
    const labelX = box.right - 13 * U;
    const labelY = box.bottom - 3 * U - 2;
    const leader = `M${mark[0].toFixed(1)} ${mark[1] + 8} V${labelY - 5} H${labelX - 7}`;
    return (
        <g>
            <Chart
                box={box}
                x={x}
                y={y}
                xTicks={[-5, 0, 5, 10, 15]}
                yTicks={[0, 0.5, 1, 1.5]}
                yFormat={(v) => (v === 0 ? "0" : v.toFixed(1))}
                xTitle="Angle of attack, degrees"
                yTitle="Section lift coefficient"
            />
            {series.map((entry, k) => (
                <g key={entry.re}>
                    <path
                        className="line"
                        d={pathFrom(entry.polar.map((p) => [x(p.alpha), y(p.cl)]))}
                    />
                    {entry.polar
                        .filter((_, i) => i % SYMBOL_STEP === 0)
                        .map((p) => (
                            <PlotSymbol
                                key={p.alpha}
                                kind={SYMBOLS[k]}
                                x={x(p.alpha)}
                                y={y(p.cl)}
                            />
                        ))}
                </g>
            ))}
            <path className="mark" d={pathFrom(tangent)} />
            <path className="mark mark--thin" d={leader} />
            <circle className="mark mark--point" cx={mark[0]} cy={mark[1]} r={6.5} />
            <text x={labelX} y={labelY}>
                {minus(`slope ${marked.dcl_dalpha.toFixed(4)} / deg`)}
            </text>
        </g>
    );
}

function DragChart({ x0, y0 }: { x0: number; y0: number }): ReactNode {
    const box = { left: x0, right: x0 + DRAG_COLS * U, top: y0, bottom: y0 + ROWS * U };
    const x = linear(0, CD_MAX, box.left, box.right);
    const y = linear(LIFT[0], LIFT[1], box.bottom, box.top);
    const inside = (p: Row): boolean => p.cd <= CD_MAX;
    return (
        <g>
            <Chart
                box={box}
                x={x}
                y={y}
                xTicks={[0, 0.01, 0.02, 0.03, 0.04]}
                yTicks={[0, 0.5, 1, 1.5]}
                xFormat={(v) => (v === 0 ? "0" : v.toFixed(2))}
                yFormat={(v) => (v === 0 ? "0" : v.toFixed(1))}
                xTitle="Section drag coefficient"
                yTitle="Section lift coefficient"
            />
            {series.map((entry, k) => (
                <g key={entry.re}>
                    <path
                        className="line"
                        d={pathFrom(entry.polar.filter(inside).map((p) => [x(p.cd), y(p.cl)]))}
                    />
                    {entry.polar
                        .filter((p, i) => i % SYMBOL_STEP === 0 && inside(p))
                        .map((p) => (
                            <PlotSymbol key={p.alpha} kind={SYMBOLS[k]} x={x(p.cd)} y={y(p.cl)} />
                        ))}
                </g>
            ))}
            <circle className="mark mark--point" cx={x(marked.cd)} cy={y(marked.cl)} r={6.5} />
        </g>
    );
}

/* ── The figure ── */

const profileLabel = `Airfoil profile of the quickstart section. At the marked condition transition is at ${marked.top_xtr.toFixed(2)} chord on the upper surface and ${marked.bot_xtr.toFixed(2)} chord on the lower surface.`;
const liftLabel = `Lift curve at three Reynolds numbers. At ${marked.alpha} degrees and a Reynolds number of one million the lift coefficient is ${marked.cl.toFixed(3)} and its slope is ${marked.dcl_dalpha.toFixed(4)} per degree.`;
const dragLabel = `Drag polar at three Reynolds numbers. At ${marked.alpha} degrees and a Reynolds number of one million the drag coefficient is ${marked.cd.toFixed(5)}.`;

// Five rows of the sweep around the marked angle, one degree apart.
const tableRows = [-8, -4, 0, 4, 8].map((offset) => rows[markedIndex + offset]);

export default function PolarFigure({
    variant = "column",
    caption,
    className,
    children,
}: {
    /** "sheet" is the page-wide home page figure; "column" fits the reading column. */
    variant?: "sheet" | "column";
    caption?: ReactNode;
    className?: string;
    /** On the sheet, the text lettered under the name in the title block. */
    children?: ReactNode;
}): ReactNode {
    const sheet = variant === "sheet";

    const titleBlock = (
        <div className={styles.block}>
            <div className={`${styles.cell} ${styles.describe}`}>
                Section characteristics of the quickstart airfoil, NeuralFoil {data.model} network
            </div>
            <div className={`${styles.cell} ${styles.key}`}>
                <div>Reynolds number, millions</div>
                <ul>
                    {series.map((entry, k) => (
                        <li key={entry.re}>
                            <svg className="bora-plot" viewBox="0 0 16 16" aria-hidden="true">
                                <PlotSymbol kind={SYMBOLS[k]} x={8} y={8} r={5.2} />
                            </svg>
                            {millions(entry.re)}
                        </li>
                    ))}
                </ul>
            </div>
            <div className={`${styles.cell} ${styles.condition}`}>
                <div>Values at the mark</div>
                <strong>
                    {marked.alpha.toFixed(1)}°, Reynolds number {millions(series[MARKED_SERIES].re)}{" "}
                    million
                </strong>
            </div>
            <dl className={`${styles.cell} ${styles.readouts}`}>
                <div>
                    <dt>Lift</dt>
                    <dd>{minus(marked.cl.toFixed(3))}</dd>
                </div>
                <div>
                    <dt>Drag</dt>
                    <dd>{marked.cd.toFixed(5)}</dd>
                </div>
                <div>
                    <dt>Moment</dt>
                    <dd>{minus(marked.cm.toFixed(3))}</dd>
                </div>
                <div>
                    <dt>Lift / drag</dt>
                    <dd>{minus((marked.cl / marked.cd).toFixed(1))}</dd>
                </div>
                <div>
                    <dt>Slope / deg</dt>
                    <dd>{minus(marked.dcl_dalpha.toFixed(4))}</dd>
                </div>
                <div>
                    <dt>Confidence</dt>
                    <dd>{marked.confidence.toFixed(3)}</dd>
                </div>
            </dl>
        </div>
    );

    return (
        <figure
            className={`bora-figure ${styles.figure} ${sheet ? styles.sheet : ""} ${className ?? ""}`}
        >
            <div className={styles.stage}>
                {sheet && (
                    // No viewBox: one unit is one pixel, so a wider window shows more sheet.
                    <svg className={`bora-plot ${styles.wide}`} width="100%" height={55 * U}>
                        <Sheet width="100%" height="100%" />
                        <g role="img" aria-label={profileLabel}>
                            <Profile x0={6 * U} yc={14 * U} chord={60 * U} />
                        </g>
                        <text x={6 * U} y={21 * U - 2}>
                            Quickstart section. Ticks: transition at the marked condition, in chord
                            fractions.
                        </text>
                        <g role="img" aria-label={liftLabel}>
                            <LiftChart x0={10 * U} y0={23 * U} />
                        </g>
                        <g role="img" aria-label={dragLabel}>
                            <DragChart x0={44 * U} y0={23 * U} />
                        </g>
                    </svg>
                )}

                <div className={`bora-panel ${styles.panels}`}>
                    <svg
                        className={`bora-plot ${styles.profile}`}
                        viewBox={`0 0 ${56 * U} ${16 * U}`}
                        role="img"
                        aria-label={profileLabel}
                    >
                        <Sheet width={56 * U} height={16 * U} />
                        <Profile x0={3 * U} yc={9 * U} chord={50 * U} />
                    </svg>
                    {/* The same drawing at a scale whose lettering survives a phone. */}
                    <svg
                        className={`bora-plot ${styles.profile} ${styles.profileSmall}`}
                        viewBox={`0 0 ${30 * U} ${12 * U}`}
                        aria-hidden="true"
                    >
                        <Sheet width={30 * U} height={12 * U} />
                        <Profile x0={2.5 * U} yc={7 * U} chord={25 * U} />
                    </svg>
                    <svg
                        className={`bora-plot ${styles.lift}`}
                        viewBox={`0 0 ${33 * U} ${32 * U}`}
                        role="img"
                        aria-label={liftLabel}
                    >
                        <Sheet width={33 * U} height={32 * U} />
                        <LiftChart x0={6 * U} y0={2 * U} />
                    </svg>
                    <svg
                        className={`bora-plot ${styles.drag}`}
                        viewBox={`0 0 ${29 * U} ${32 * U}`}
                        role="img"
                        aria-label={dragLabel}
                    >
                        <Sheet width={29 * U} height={32 * U} />
                        <DragChart x0={7 * U} y0={2 * U} />
                    </svg>
                </div>

                {sheet ? (
                    <div className={`${styles.block} ${styles.nameplate}`}>
                        <div className={styles.name} aria-hidden="true">
                            Bora
                        </div>
                        {children && <div className={styles.lede}>{children}</div>}
                    </div>
                ) : (
                    titleBlock
                )}
            </div>

            {!sheet && (
                <div className={styles.scroll}>
                    <table className={`bora-table ${styles.table}`}>
                        <thead>
                            <tr>
                                <th scope="col">α,&nbsp;deg</th>
                                <th scope="col">Lift</th>
                                <th scope="col">Drag</th>
                                <th scope="col">Moment</th>
                                <th scope="col">Lift / drag</th>
                                <th scope="col">Transition, upper</th>
                                <th scope="col">Transition, lower</th>
                                <th scope="col">Slope / deg</th>
                                <th scope="col">Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            {tableRows.map((r) => (
                                <tr
                                    key={r.alpha}
                                    className={r === marked ? styles.marked : undefined}
                                >
                                    <th scope="row">{minus(r.alpha.toFixed(2))}</th>
                                    <td>{minus(r.cl.toFixed(4))}</td>
                                    <td>{r.cd.toFixed(5)}</td>
                                    <td>{minus(r.cm.toFixed(4))}</td>
                                    <td>{minus((r.cl / r.cd).toFixed(1))}</td>
                                    <td>{r.top_xtr.toFixed(3)}</td>
                                    <td>{r.bot_xtr.toFixed(3)}</td>
                                    <td>{minus(r.dcl_dalpha.toFixed(5))}</td>
                                    <td>{r.confidence.toFixed(3)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            <figcaption className={sheet ? styles.sheetCaption : undefined}>
                {caption ?? (
                    <>
                        The quickstart airfoil at three Reynolds numbers: profile, lift curve and
                        drag polar, with the rows of the sweep around the marked angle. The ticks on
                        the profile give the transition location at the marked condition, in chord
                        fractions. The heavy line on the lift curve is the analytic slope returned
                        with that one evaluation. One small square is 1° across, 0.1 in lift and
                        0.002 in drag.
                    </>
                )}
            </figcaption>
        </figure>
    );
}
