import type { ReactNode } from "react";

/** A numbered caption set above the table that follows it. */
export default function TableCaption({ children }: { children: ReactNode }): ReactNode {
    return <div className="bora-table-caption">{children}</div>;
}
