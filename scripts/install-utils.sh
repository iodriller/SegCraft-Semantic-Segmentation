#!/usr/bin/env bash

install_init() {
  install_root=$1
  install_product=$2
  install_state="$install_root/.setup"
  install_log="$install_state/install.log"
  install_lock_dir="$install_state/install.lock"
  install_has_lock=0
  mkdir -p "$install_state"
  if [ -f "$install_log" ] && [ "$(wc -c <"$install_log")" -gt 1048576 ]; then
    mv -f -- "$install_log" "$install_log.1"
  fi
  install_note "start pid=$$ platform=$(uname -s)"
}

install_note() {
  [ -n "${install_log:-}" ] || return 0
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$install_log"
}

install_lock() {
  local waited=0 owner=""
  while ! mkdir "$install_lock_dir" 2>/dev/null; do
    owner=$(cat "$install_lock_dir/pid" 2>/dev/null || true)
    if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
      rm -rf -- "$install_lock_dir"
      continue
    fi
    [ "$waited" -lt 240 ] || { echo "Another $install_product installation is still running. Lock: $install_lock_dir" >&2; return 1; }
    sleep 0.25
    waited=$((waited + 1))
  done
  printf '%s\n' "$$" >"$install_lock_dir/pid"
  install_has_lock=1
  install_note "lock acquired"
}

install_unlock() {
  [ "${install_has_lock:-0}" -eq 1 ] || return 0
  rm -rf -- "$install_lock_dir"
  install_has_lock=0
  install_note "lock released"
}

install_on_error() {
  local code=$1 line=$2
  trap - ERR
  install_note "failure exit=$code line=$line"
  printf 'Setup log: %s\n' "$install_log" >&2
  install_unlock
  exit "$code"
}

install_on_signal() {
  local code=$1 signal=$2
  trap - EXIT ERR INT TERM
  install_note "interrupted signal=$signal"
  install_unlock
  exit "$code"
}

install_enable_traps() {
  trap 'install_on_error "$?" "$LINENO"' ERR
  trap 'install_unlock' EXIT
  trap 'install_on_signal 130 INT' INT
  trap 'install_on_signal 143 TERM' TERM
}

install_download() {
  local url=$1 destination=$2 label=${3:-download} partial="${2}.partial" method attempt code http
  mkdir -p -- "$(dirname -- "$destination")"
  rm -f -- "$partial"
  for attempt in 1 2 3; do
    if command -v curl >/dev/null 2>&1 && { [ "$attempt" -ne 2 ] || ! command -v wget >/dev/null 2>&1; }; then
      method=curl
      set +e
      http=$(curl --location --silent --show-error --output "$partial" --write-out '%{http_code}' "$url")
      code=$?
      set -e
      if [ "$code" -eq 0 ] && [ -s "$partial" ]; then mv -f -- "$partial" "$destination"; install_note "$label completed with curl"; return 0; fi
      case "$http" in 400|401|402|403|404|405|406|409|410|411|412|413|414|415|416|417|422) echo "$label failed with HTTP $http" >&2; return 1 ;; esac
    elif command -v wget >/dev/null 2>&1; then
      method=wget
      if wget --quiet --timeout=30 --output-document="$partial" "$url" && [ -s "$partial" ]; then mv -f -- "$partial" "$destination"; install_note "$label completed with wget"; return 0; fi
    else
      echo "curl or wget is required for $label" >&2
      return 1
    fi
    install_note "$label transient failure attempt=$attempt method=$method"
    rm -f -- "$partial"
    [ "$attempt" -eq 3 ] || sleep $((1 << (attempt - 1)))
  done
  echo "$label failed after 3 bounded attempts" >&2
  return 1
}

install_retry() {
  local label=$1; shift
  local attempt error_file attempts="${INSTALL_RETRY_ATTEMPTS:-3}"
  case "$attempts" in 1|2|3|4|5) ;; *) attempts=3 ;; esac
  error_file=$(mktemp)
  attempt=1
  while [ "$attempt" -le "$attempts" ]; do
    : >"$error_file"
    if "$@" 2> >(tee "$error_file" >&2); then rm -f -- "$error_file"; return 0; fi
    if grep -Eqi 'checksum|hash mismatch|permission|access denied|unauthorized|forbidden|not found|404|unsupported|invalid argument|license' "$error_file" ||
       ! grep -Eqi 'timed? out|temporar|connection|name resolution|network|reset by peer|429|408|500|502|503|504|service unavailable|gateway' "$error_file"; then
      rm -f -- "$error_file"
      return 1
    fi
    install_note "$label transient failure attempt=$attempt"
    [ "$attempt" -eq "$attempts" ] || sleep $((1 << (attempt - 1)))
    attempt=$((attempt + 1))
  done
  rm -f -- "$error_file"
  return 1
}

install_require_space() {
  local path=$1 required_gb=$2 available_kb required_kb
  available_kb=$(df -Pk "$path" | awk 'NR==2 {print $4}')
  required_kb=$((required_gb * 1024 * 1024))
  [ -n "$available_kb" ] && [ "$available_kb" -ge "$required_kb" ] || {
    echo "Insufficient disk space for $install_product. Required: ${required_gb} GB." >&2
    return 1
  }
  install_note "disk available_kb=$available_kb required_kb=$required_kb"
}

install_complete() {
  install_note "bootstrap complete"
  install_unlock
}
