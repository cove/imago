#!/usr/bin/env nu

cd $env.FILE_PWD

# Parse and load .env, expanding $HOME / ${HOME} as bash would
let home_path = $env.HOME
open --raw .env
| lines
| where {|line| ($line | str trim) != "" and not ($line | str starts-with "#")}
| each {|line|
    let parts = $line | split row "="
    let key = $parts | first | str trim
    let val = ($parts | skip 1 | str join "=") | str replace -r '\$\{?HOME\}?' $home_path
    {($key): $val}
  }
| reduce --fold {} {|it, acc| $acc | merge $it}
| load-env

# On Windows, find binary in WinGet packages if not on PATH
let server_bin = if $nu.os-info.name == "windows" and (which $env.LLAMA_SERVER_BIN | is-empty) {
  let pkg_root = [$env.LOCALAPPDATA, "Microsoft", "WinGet", "Packages"] | path join
  glob $"($pkg_root)/**/*($env.LLAMA_SERVER_BIN).exe" | first
} else {
  $env.LLAMA_SERVER_BIN
}

let model = [$env.MODEL_DIR, $env.MODEL_FILE] | path join
let mmproj = [$env.MODEL_DIR, $env.MMPROJ_FILE] | path join
let filter = ($env.FILE_PWD | path join "filter.py")
let logs_dir = ($env.FILE_PWD | path join ".." "logs")
let llama_log = ($logs_dir | path join "llama.log")
mkdir $logs_dir

(
  ^$server_bin
    --model $model
    --mmproj $mmproj
    --alias $env.MODEL_ALIAS
    --host $env.HOST
    --port $env.PORT
    --parallel $env.PARALLEL
    --ctx-size $env.CTX_SIZE
    --cache-ram $env.CACHE_RAM
    --repeat-penalty $env.REPEAT_PENALTY
    --timeout 600
    --verbose
    e+o>| python3 $filter | tee { save --append $llama_log }
)

if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
