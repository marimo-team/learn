#!/usr/bin/env bash
for nb in $*
do
	cd $(dirname $nb)
	if ! output=$(uv run $(basename $nb) 2>&1); then
		echo "=== $nb ==="
		echo "$output"
		echo
	fi
	cd $OLDPWD
done
