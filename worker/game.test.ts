import { describe, expect, it } from 'vitest';
import { encodeBid, isHigherBid } from './game';

describe('Perudo Cloudflare game core', () => {
  it('keeps the existing action encoding contract', () => {
    expect(encodeBid(1, 1)).toBe(0);
    expect(encodeBid(2, 6)).toBe(11);
  });

  it('applies ones bid rules', () => {
    expect(isHigherBid(3, 2, [1, 1])).toBe(true);
    expect(isHigherBid(2, 2, [1, 1])).toBe(false);
    expect(isHigherBid(1, 1, [2, 4])).toBe(true);
  });
});
