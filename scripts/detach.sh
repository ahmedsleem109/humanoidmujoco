#!/usr/bin/env bash
# Detach a long-running command from the calling session, correctly.
#
#   detach.sh <LOGFILE> <command> [args...]
#
# Two things have to be right or the child dies when `wsl.exe` returns:
#
#   1. setsid  -- a new session, so closing the caller's console cannot deliver
#                 SIGHUP to it;
#   2. stdout AND stderr redirected to a file, plus stdin from /dev/null.
#      Without this the child writes to a closed pipe on its first `echo` and
#      is killed by SIGPIPE. That is not hypothetical: it killed the
#      supervisor rehearsal while the training process it had launched -- which
#      does redirect -- carried on running, which looks exactly like the
#      supervisor "finishing".
#
# Prints the child PID so the caller can verify it is actually alive rather
# than trusting that the launch returned 0.
set -u

LOGFILE=${1:?logfile required}
shift
mkdir -p "$(dirname "$LOGFILE")"
: > "$LOGFILE"

setsid nohup "$@" </dev/null >>"$LOGFILE" 2>&1 &
CHILD=$!

sleep 2
if kill -0 "$CHILD" 2>/dev/null; then
  echo "detached pid=$CHILD log=$LOGFILE"
else
  echo "FAILED: child exited immediately; log follows"
  cat "$LOGFILE"
  exit 1
fi
