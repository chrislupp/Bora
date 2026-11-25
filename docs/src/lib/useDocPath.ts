import { useVersions } from "@docusaurus/plugin-content-docs/client";

/**
 * Path of a doc by id, for links from pages outside the docs (the home page).
 *
 * Prefers the latest released version and falls back to the unreleased
 * ("current") docs, so a page that only exists on `develop` still resolves
 * and the link moves to the released copy once a release includes it.
 */
export function useDocPath(id: string): string {
    const versions = useVersions("default");
    const latest = versions.find((version) => version.isLast);
    const current = versions.find((version) => version.name === "current");
    for (const version of [latest, current]) {
        const doc = version?.docs.find((candidate) => candidate.id === id);
        if (doc) {
            return doc.path;
        }
    }
    return (latest ?? versions[0]).path;
}
