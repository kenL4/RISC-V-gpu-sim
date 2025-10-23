#pragma once

#include <algorithm>
#include <bits/stdc++.h>
#include <capstone/capstone.h>
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "config.hpp"
#include "parser.hpp"

/*
 * Prints a generic message with an associated timestamp
 */
void debug_log(std::string message);

/*
 * Prints a named message with an associated timestamp
 */
void log(std::string name, std::string message);