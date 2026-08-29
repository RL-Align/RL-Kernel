# Contributing to RL-Kernel

Thank you for your interest in contributing to RL-Kernel. Bug reports, feature
requests, documentation improvements, and code contributions are all welcome.

- For bugs and feature requests, please
  [open an issue](https://github.com/RL-Align/RL-Kernel/issues) first.
- For code changes, open a pull request against `main` and keep it focused on
  a single topic.
- For questions and discussion, reach the community on
  [Slack](https://rl-align.slack.com) or [WeChat](./docs/community/wechat.md).

## Sign your work (DCO)

Every commit in a pull request must carry a `Signed-off-by:` trailer, per the
[Developer Certificate of Origin](./DCO) (DCO), version 1.1. The DCO check is
enforced as a required status check on every pull request.

By signing off, you certify that you wrote the change or otherwise have the
right to submit it under the project's open source license.

Sign a new commit:

```bash
git commit -s
```

This appends `Signed-off-by: Your Name <your.email@example.com>` to the commit
message.

Repair commits you have already pushed (for example, when the DCO check
reports a missing sign-off). If you work from a fork, rebase onto the upstream
repository:

```bash
git remote add upstream https://github.com/RL-Align/RL-Kernel.git  # once, if missing
git fetch upstream
git rebase --signoff upstream/main
git push --force-with-lease
```

If you cloned the project repository directly, rebase onto `origin/main`
instead:

```bash
git fetch origin
git rebase --signoff origin/main
git push --force-with-lease
```

The DCO check validates the **email address**: the `Signed-off-by:` email
must match the commit author email. Using `git commit -s` guarantees this,
since it signs with your configured `user.name` and `user.email`:

```text
Author:         Your Name <your.email@example.com>
Signed-off-by:  Your Name <your.email@example.com>
```

Sign with a name that identifies you to the project — your real name, or a
name or handle linked to your GitHub account. The automated check compares
only the email; the name is what maintainers use to recognize who made the
certification.

### Email matching

If you contribute from several machines, keep `git config user.email`
consistent, and make sure the address is associated with your GitHub account
so commits are attributed to you and the DCO check passes.

### Signing policies

- `Co-authored-by:` trailers do not replace `Signed-off-by:`; each commit must
  be signed off by its own author.
- Commits produced by bots or other automation must still be signed off by the
  human who submits them. The submitting human is responsible for the change.
- AI-assisted contributions are welcome. The human submitter signs off and
  remains responsible for the correctness and licensing of the contribution.

## Pull requests opened before the DCO check was enabled

For pull requests created before the DCO check became a required status check,
maintainers will leave a one-time comment pointing to this guide. Authors are
asked to add the missing sign-offs within a four-week grace period. Pull
requests that still fail the DCO check after the grace period may be closed,
with an invitation to reopen once the commits are signed.

## Development guide

For setting up a development environment, running tests, and documentation
conventions, see the [developer guide](./docs/contributing/README.md).
