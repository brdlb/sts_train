import React, { useMemo } from 'react';

const BackgroundStars: React.FC = () => {
  const stars = useMemo(() => {
    const starArray = [];
    for (let i = 0; i < 100; i++) {
      const size = Math.random() * 2 + 1; // 1px to 3px
      const duration = Math.random() * 3 + 2; // 2s to 5s
      const delay = Math.random() * 5; // 0s to 5s
      const top = Math.random() * 100;
      const left = Math.random() * 100;
      starArray.push(
        <div
          key={i}
          className="star"
          style={{
            width: `${size}px`,
            height: `${size}px`,
            top: `${top}%`,
            left: `${left}%`,
            animationDuration: `${duration}s`,
            animationDelay: `${delay}s`,
          }}
        />
      );
    }
    return starArray;
  }, []);

  return (
    <div className="fixed inset-0 w-full h-full pointer-events-none z-0 overflow-hidden">
      {stars}
    </div>
  );
};

export default BackgroundStars;
