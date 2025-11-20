// Action encoding utilities

export function encode_bid(quantity: number, value: number): number {
  // Action encoding: 0=challenge, 1=believe, 2+=bid
  return 2 + (quantity - 1) * 6 + (value - 1);
}

