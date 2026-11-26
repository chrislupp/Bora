import type { ReactNode } from "react";
import { useDocsVersion } from "@docusaurus/plugin-content-docs/client";
import CodeBlock from "@theme/CodeBlock";

/**
 * The `docker run` command for the image tag that matches the docs being read:
 * `next` (the develop branch) in the unreleased docs, the release number in a
 * release's docs, so a versioned snapshot never points at a moving tag.
 */
export function useDockerTag(): string {
    const version = useDocsVersion();
    return version.version === "current" ? "next" : version.version;
}

export default function DockerRun(): ReactNode {
    const tag = useDockerTag();
    return (
        <CodeBlock language="bash">
            {`docker run -d --rm --name bora -p 50051:50051 chrislupp/bora:${tag}`}
        </CodeBlock>
    );
}
