# Security policy

## Supported versions

Until the first stable release, security fixes are applied to the latest revision of
the `main` branch. After versioned releases begin, the latest release will receive
security fixes. Older development snapshots are not supported.

## Reporting a vulnerability

Please do not report suspected vulnerabilities in a public issue.

Use GitHub's private vulnerability-reporting feature when it is available for the
repository. Otherwise, contact Yannik Heizmann at
`yheizman@stud.hs-offenburg.de` or Harald Hoppe at
`harald.hoppe@hs-offenburg.de`. Include:

- the affected version or commit;
- the vulnerable component and execution path;
- steps or a minimal example that reproduce the issue;
- the expected impact; and
- any known mitigations.

Do not include API tokens, credentials, private model repositories, or sensitive data
in a report. Reports will be acknowledged as soon as practical. A remediation timeline
will depend on the severity and complexity of the issue.

## Scope

Reports concerning MoLAG's source code, packaging, artefact loading, command-line
workflows, and documented integrations are in scope. Vulnerabilities in third-party
dependencies should normally be reported to their maintainers, unless MoLAG uses the
dependency in a way that creates a project-specific vulnerability.
