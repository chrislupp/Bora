import type { ReactNode } from "react";

const W = 600;
const H = 508;

function Box({
    x,
    y,
    w,
    h = 44,
    lines,
}: {
    x: number;
    y: number;
    w: number;
    h?: number;
    lines: string[];
}): ReactNode {
    const first = y + h / 2 + 5 - ((lines.length - 1) * 16) / 2;
    return (
        <g>
            <rect className="frame" x={x} y={y} width={w} height={h} />
            {lines.map((line, i) => (
                <text
                    key={line}
                    className="plain"
                    x={x + w / 2}
                    y={first + i * 16}
                    textAnchor="middle"
                >
                    {line}
                </text>
            ))}
        </g>
    );
}

const arrow = (x: number, y0: number, y1: number): string =>
    `M${x} ${y0} V${y1} M${x - 4} ${y1 - 7} L${x} ${y1} L${x + 4} ${y1 - 7}`;

/** The five steps of one analysis, top to bottom. */
export default function Pipeline(): ReactNode {
    return (
        <figure className="bora-figure">
            <div className="bora-scroll">
                <svg
                    className="bora-plot"
                    viewBox={`0 0 ${W} ${H}`}
                    style={{ maxWidth: `${W}px`, minWidth: "32rem" }}
                    role="img"
                    aria-label="Flow of one analysis: 23 inputs are encoded to 25 values; the network is evaluated for the airfoil as given and for its mirror image; a distance term is subtracted from the confidence of each; the two results are averaged; and the average is decoded to the physical outputs."
                >
                    <Box x={110} y={8} w={380} lines={["23 inputs: shape and operating point"]} />
                    <path className="tick" d={arrow(300, 52, 86)} />
                    <text x={312} y={74}>
                        1. encode
                    </text>
                    <Box x={200} y={86} w={200} lines={["25 encoded values"]} />

                    <path className="tick" d="M250 130 V150 H150 V186 M146 179 L150 186 L154 179" />
                    <path
                        className="tick line--dashed"
                        d="M350 130 V150 H450 V186 M446 179 L450 186 L454 179"
                    />
                    <text x={138} y={172} textAnchor="end">
                        as given
                    </text>
                    <text x={462} y={172}>
                        mirror image
                    </text>

                    <Box x={40} y={186} w={220} lines={["2. network"]} />
                    <Box x={340} y={186} w={220} lines={["2. network"]} />

                    <path className="tick" d={arrow(150, 230, 264)} />
                    <path className="tick line--dashed" d={arrow(450, 230, 264)} />
                    <Box x={40} y={264} w={220} lines={["3. subtract distance term"]} />
                    <Box x={340} y={264} w={220} lines={["3. subtract distance term"]} />

                    <path className="tick" d="M150 308 V330 H250 V352 M246 345 L250 352 L254 345" />
                    <path
                        className="tick line--dashed"
                        d="M450 308 V330 H350 V352 M346 345 L350 352 L354 345"
                    />
                    <text x={462} y={334}>
                        map back
                    </text>

                    <Box x={200} y={352} w={200} lines={["4. average the two"]} />
                    <path className="tick" d={arrow(300, 396, 440)} />
                    <text x={312} y={422}>
                        5. decode
                    </text>
                    <Box
                        x={70}
                        y={440}
                        w={460}
                        h={60}
                        lines={[
                            "lift, drag, moment, transition, confidence,",
                            "momentum thickness, shape factor, edge velocity",
                        ]}
                    />
                </svg>
            </div>
            <figcaption>
                One analysis. The dashed branch evaluates the mirror image, the shape flipped top to
                bottom at the negated angle of attack, which enforces the symmetry exactly.
            </figcaption>
        </figure>
    );
}
