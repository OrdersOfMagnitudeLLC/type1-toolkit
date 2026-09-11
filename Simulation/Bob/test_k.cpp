// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "bob.hpp"
#include <iostream>
#include <cassert>

int main() {
    // Create BobIndex with K=2 properties, N=5 materials
    BobIndex index(2, 5);
    
    // Add 5 fake materials with 2 properties each
    // Material 0: props = [10.0, 20.0]
    index.add_material(0, {10.0, 20.0});
    
    // Material 1: props = [15.0, 25.0]
    index.add_material(1, {15.0, 25.0});
    
    // Material 2: props = [30.0, 40.0]
    index.add_material(2, {30.0, 40.0});
    
    // Material 3: props = [35.0, 45.0]
    index.add_material(3, {35.0, 45.0});
    
    // Material 4: props = [50.0, 60.0]
    index.add_material(4, {50.0, 60.0});
    
    // Build indices
    index.build_indices();
    
    // Query with bounds that should return materials 1, 2, 3
    // Property 0: 12.0 to 40.0 (matches materials 1, 2, 3)
    // Property 1: 22.0 to 50.0 (matches materials 1, 2, 3)
    std::vector<PropBounds> bounds = {
        {12.0, 40.0},  // Property 0 bounds
        {22.0, 50.0}   // Property 1 bounds
    };
    
    auto results = index.query(bounds);
    
    // Print results
    std::cout << "Query bounds:\n";
    std::cout << "  Property 0: [" << bounds[0].min_val << ", " << bounds[0].max_val << "]\n";
    std::cout << "  Property 1: [" << bounds[1].min_val << ", " << bounds[1].max_val << "]\n";
    std::cout << "Results: ";
    for (size_t idx : results) {
        std::cout << idx << " ";
    }
    std::cout << "\n";
    
    // Expected: materials 1, 2, 3 (indices 1, 2, 3)
    // Material 0: [10, 20] - fails property 0 (10 < 12)
    // Material 1: [15, 25] - passes both
    // Material 2: [30, 40] - passes both
    // Material 3: [35, 45] - passes both
    // Material 4: [50, 60] - fails property 0 (50 > 40)
    
    std::cout << "Expected: 1 2 3\n";
    
    // Assert we got exactly 3 results
    assert(results.size() == 3);
    
    // Assert the results are exactly 1, 2, 3 (in any order)
    std::set<size_t> result_set(results.begin(), results.end());
    assert(result_set.count(1) == 1);
    assert(result_set.count(2) == 1);
    assert(result_set.count(3) == 1);
    
    std::cout << "Test passed!\n";
    
    return 0;
}
