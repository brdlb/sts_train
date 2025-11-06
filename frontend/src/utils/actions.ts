// Action encoding utilities

export function encode_bid(quantity: number, value: number): number {
  // Action encoding: 0=challenge, 1=believe, 2+=bid
  return 2 + (quantity - 1) * 6 + (value - 1);
}

export function decode_bid(action: number): [number, number] {
  // Decode bid from action code
  const bidAction = action - 2;
  const quantity = Math.floor(bidAction / 6) + 1;
  const value = (bidAction % 6) + 1;
  return [quantity, value];
}

