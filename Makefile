BUILD_DIR = build
EXEC = RISCVGpuSim
TEST_EXEC = unit_tests

# Default target
all: $(BUILD_DIR)/Makefile
	@$(MAKE) --no-print-directory -C $(BUILD_DIR)

# Ensure build directory exists and run CMake
$(BUILD_DIR)/Makefile: CMakeLists.txt
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. > /dev/null

# Build just the unit_tests target
unit_tests: $(BUILD_DIR)/Makefile
	@$(MAKE) --no-print-directory -C $(BUILD_DIR) unit_tests

# Build and run unit tests
test: unit_tests
	@cd $(BUILD_DIR) && ./$(TEST_EXEC)

# Clean build directory
clean:
	@rm -rf $(BUILD_DIR)

.PHONY: all unit_tests test clean run
