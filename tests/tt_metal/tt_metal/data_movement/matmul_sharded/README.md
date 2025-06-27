# Matmul Sharded Data Movement Tests

This test suite implements tests that measure the performance (i.e. bandwidth) of matmul sharded transactions between Tensix cores.

## Test Flow

It does not check pcc as the afformentioned test does this.

The matmul sharded patterns are the exact ones as gotten from the base tests, as such this is a directed test is is not general.

## Test Parameters
| Parameter                 | Data Type                          | Description |
| ------------------------- | ---------------------              | ----------- |
| test_id                   | uint32_t                           | Test id for signifying different test cases. |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. Matmul Sharded
