#include "test_pipeline_scheduler.hpp"
#include "gpu/pipeline_warp_scheduler.hpp"
#include <cassert>
#include <iostream>
#include <vector>

void test_warp_scheduler() {
  std::cout << "Running test_warp_scheduler..." << std::endl;

  WarpScheduler scheduler(32, 4, 0x1000);
  PipelineLatch input, output;
  output.updated = false;
  scheduler.set_latches(&input, &output);
  scheduler.set_debug(false);

  // 1. Initial Sequence (Expected: 0, 1, 2, 3)
  std::cout << "  Phase 1: Round Robin" << std::endl;
  for (int i = 0; i < 8; ++i) {
    scheduler.execute();
    if (output.updated) {
      std::cout << "    Cycle " << i << ": Got " << output.warp->warp_id
                << std::endl;
      scheduler.insert_warp(output.warp);
      output.updated = false;
    }
  }

  // 2. Suspend 1
  std::cout << "  Phase 2: Suspend Warp 1" << std::endl;
  // Based on loop 8, we did 0,1,2,3,0,1,2,3.
  // Next is 0.

  // Run until we get 1
  Warp *warp1 = nullptr;
  for (int i = 0; i < 4; ++i) {
    scheduler.execute();
    assert(output.updated);
    std::cout << "    Cycle " << i << ": Got " << output.warp->warp_id
              << std::endl;
    if (output.warp->warp_id == 1) {
      warp1 = output.warp;
      warp1->suspended = true;
      std::cout << "      (Suspending 1)" << std::endl;
    }
    scheduler.insert_warp(output.warp);
    output.updated = false;
  }

  // Now 1 is suspended and re-inserted.
  // Verify 1 is skipped.
  std::cout << "  Phase 3: Verify Skipped" << std::endl;
  for (int i = 0; i < 8; ++i) {
    scheduler.execute();
    if (output.updated) {
      std::cout << "    Cycle " << i << ": Got " << output.warp->warp_id
                << std::endl;
      assert(output.warp->warp_id != 1);
      scheduler.insert_warp(output.warp);
      output.updated = false;
    } else {
      std::cout << "    Cycle " << i << ": Empty/Skipped" << std::endl;
    }
  }

  // 3. Unsuspend 1
  std::cout << "  Phase 4: Unsuspend Warp 1" << std::endl;
  if (warp1)
    warp1->suspended = false;

  // Verify 1 returns
  bool found_1 = false;
  for (int i = 0; i < 8; ++i) {
    scheduler.execute();
    if (output.updated) {
      std::cout << "    Cycle " << i << ": Got " << output.warp->warp_id
                << std::endl;
      if (output.warp->warp_id == 1)
        found_1 = true;
      scheduler.insert_warp(output.warp);
      output.updated = false;
    }
  }
  assert(found_1);

  std::cout << "test_warp_scheduler passed!" << std::endl;
}
