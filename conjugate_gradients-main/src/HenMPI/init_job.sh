#!/bin/bash

# Submit job allocation request
salloc -A p200301 --res cpudev -q dev -N 1 -t 01:00:00