

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BOT_PERSONALITIES, PLAYER_AVATAR, INITIAL_DICE_COUNT, PLAYER_COLORS } from './constants';
import { getBotDecision, shouldBotStartSpecialRound } from './services/geminiService';
import type { GameState, Player, DiceValue, Bid, BotDecision, LogEntry, BotPersonality, PlayerAnalysis } from './types';
import PlayerComponent from './components/Player';
import BidControls from './components/BidControls';
import GameLog from './components/GameLog';
import Modal from './components/Modal';
import BackgroundStars from './components/BackgroundStars';
import LogModal from './components/LogModal';

const rollDice = (count: number): DiceValue[] => {
  return Array.from({ length: count }, () => (Math.floor(Math.random() * 6) + 1) as DiceValue);
};

const shuffleArray = <T,>(array: T[]): T[] => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

const formatBid = (bid: Bid) => `${bid.quantity} x ${bid.face === 1 ? '★' : bid.face}`;

// A separate modal state for the special round decision
const SpecialRoundDecisionModal: React.FC<{ onDecision: (start: boolean) => void }> = ({ onDecision }) => {
    return (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg shadow-xl p-8 max-w-lg w-full m-4">
                <h2 className="text-3xl font-bold text-white mb-4">Объявить Special раунд?</h2>
                <div className="text-gray-300 mb-6 text-lg">
                    <p>У вас осталась одна кость! Вы можете объявить Special раунд. В этом раунде единицы не являются джокерами, и номинал ставки нельзя менять.</p>
                    <p className="text-base text-gray-400 mt-2">Вы можете сделать это только один раз за игру.</p>
                </div>
                <div className="flex justify-end space-x-4">
                    <button onClick={() => onDecision(false)} className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-8 rounded-lg transition-colors">
                        Отказаться (обычный раунд)
                    </button>
                    <button onClick={() => onDecision(true)} className="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-3 px-8 rounded-lg transition-colors">
                        Объявить Special раунд!
                    </button>
                </div>
            </div>
        </div>
    );
};

const createInitialPlayerAnalysis = (player: Player): PlayerAnalysis => ({
    id: player.id,
    firstBidBluffs: { count: 0, total: 0 },
    preRevealTendency: { bluffCount: 0, strongHandCount: 0, total: 0 },
    faceBluffPatterns: {
      '1': { bluffCount: 0, totalBids: 0 },
      '2': { bluffCount: 0, totalBids: 0 },
      '3': { bluffCount: 0, totalBids: 0 },
      '4': { bluffCount: 0, totalBids: 0 },
      '5': { bluffCount: 0, totalBids: 0 },
      '6': { bluffCount: 0, totalBids: 0 },
    },
});

export const App: React.FC = () => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modalContent, setModalContent] = useState<{ title: string; body: React.ReactNode } | null>(null);
  const [isLogModalOpen, setIsLogModalOpen] = useState(false);
  const [gamesPlayed, setGamesPlayed] = useState(0);
  
  // Refs for auto-scrolling on mobile
  const playerRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const bidControlsRef = useRef<HTMLDivElement | null>(null);
  const botTurnTimeoutRef = useRef<number | null>(null);

  // Google Analytics helper
  const trackEvent = useCallback((eventName: string, params?: Record<string, any>) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', eventName, params);
    }
  }, []);

  const clearBotTimeout = () => {
    if (botTurnTimeoutRef.current) {
      clearTimeout(botTurnTimeoutRef.current);
      botTurnTimeoutRef.current = null;
    }
  };

  const setupGame = useCallback((numberOfPlayers: number) => {
    clearBotTimeout(); // Fix: Prevent lingering bot actions from previous games.
    const humanPlayer: Omit<Player, 'color' | 'hasUsedSpecialRound'> = {
      id: 'human',
      name: 'Игрок',
      dice: rollDice(INITIAL_DICE_COUNT),
      isHuman: true,
      avatar: PLAYER_AVATAR,
    };

    const allBotPersonalities = Object.values(BOT_PERSONALITIES);
    let selectedBotPersonalities: BotPersonality[];

    // DEBUG: 50% chance for Standard Stan to be in the game. Easy to revert by removing this if/else block.
    if (Math.random() < 0.5) {
        const standardStan = BOT_PERSONALITIES.STANDARD_STAN;
        const otherBots = allBotPersonalities.filter(p => p.name !== standardStan.name);
        const shuffledOtherBots = shuffleArray(otherBots);
        const otherSelectedBots = shuffledOtherBots.slice(0, numberOfPlayers - 2);
        selectedBotPersonalities = shuffleArray([standardStan, ...otherSelectedBots]);
    } else {
        const shuffledBots = shuffleArray(allBotPersonalities);
        selectedBotPersonalities = shuffledBots.slice(0, numberOfPlayers - 1);
    }

    const bots: Omit<Player, 'color' | 'hasUsedSpecialRound'>[] = selectedBotPersonalities.map((p, i) => ({
      id: `bot_${i}`,
      name: p.name,
      dice: rollDice(INITIAL_DICE_COUNT),
      isHuman: false,
      personality: p,
      avatar: p.avatar,
    }));

    const uncoloredPlayers = [humanPlayer, ...bots];
    const shuffledPlayers = shuffleArray(uncoloredPlayers).map((player, index) => ({
      ...player,
      color: PLAYER_COLORS[index % PLAYER_COLORS.length],
      hasUsedSpecialRound: false, // Initialize special round status
    }));

    const initialPlayerAnalysis: Record<string, PlayerAnalysis> = {};
    shuffledPlayers.forEach(player => {
        initialPlayerAnalysis[player.id] = createInitialPlayerAnalysis(player);
    });

    // Отслеживаем начало новой игры
    if (gamesPlayed === 0) {
      trackEvent('session_start', {
        players_count: numberOfPlayers
      });
    }
    
    trackEvent('game_start', {
      game_number: gamesPlayed + 1,
      players_count: numberOfPlayers
    });

    setGameState({
      players: shuffledPlayers,
      currentPlayerIndex: 0,
      currentBid: null,
      gamePhase: 'bidding',
      gameLog: [{ message: `Новая игра началась! Удачи. Игроков: ${numberOfPlayers}.` }, { message: `--- Новый раунд #1 ---` }],
      roundStarterIndex: 0,
      lastBidderIndex: null,
      challengeType: null,
      roundBidHistory: [],
      roundOutcome: null,
      isSpecialRound: false,
      roundNumber: 1,
      playerAnalysis: initialPlayerAnalysis,
    });
  }, [gamesPlayed, trackEvent]);
  
  useEffect(() => {
    // Start the game automatically on first load.
    const randomNumPlayers = Math.floor(Math.random() * 3) + 3; // 3, 4, or 5
    setupGame(randomNumPlayers);
  }, [setupGame]);


  const addToLog = (message: string, playerId?: string) => {
    const newEntry: LogEntry = { message, playerId };
    setGameState(prev => prev ? { ...prev, gameLog: [...prev.gameLog, newEntry] } : null);
  };

  const nextTurn = () => {
    setIsLoading(false);
    setGameState(prev => {
      if (!prev) return null;
      let nextIndex = (prev.currentPlayerIndex + 1) % prev.players.length;
      while(prev.players[nextIndex].dice.length === 0) {
        nextIndex = (nextIndex + 1) % prev.players.length;
      }
      return { ...prev, currentPlayerIndex: nextIndex };
    });
  };
  
  const handlePlaceBid = (bid: Bid) => {
    if (!gameState) return;
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    
    const isFirstBidOfSpecialRound = gameState.isSpecialRound && !gameState.currentBid;
    let message: string;
    
    if (isFirstBidOfSpecialRound) {
        message = `объявляет: "Играю Special раунд на <strong>${formatBid(bid)}</strong>!"`;
    } else {
        message = `ставит <strong>${formatBid(bid)}</strong>.`;
    }
    addToLog(message, currentPlayer.id);


    setGameState(prev => {
        if (!prev) return null;
        
        return {
            ...prev,
            currentBid: bid,
            lastBidderIndex: prev.currentPlayerIndex,
            roundBidHistory: [...prev.roundBidHistory, { bid: bid, bidderId: currentPlayer.id }],
        };
    });
    nextTurn();
  };
  
  const handleChallenge = (type: 'dudo' | 'calza') => {
    if (!gameState || !gameState.currentBid || gameState.lastBidderIndex === null) return;

    const challenger = gameState.players[gameState.currentPlayerIndex];
    const bidder = gameState.players[gameState.lastBidderIndex];
    const challengeText = type === 'dudo' ? '"Не верю!"' : '"Верю!" (Точно!)';
    addToLog(`говорит ${challengeText} на ставку <strong>${bidder.name}</strong> ${formatBid(gameState.currentBid)}.`, challenger.id);

    setGameState(prev => prev ? { ...prev, gamePhase: 'reveal', challengeType: type } : null);

    setTimeout(() => {
        resolveChallenge();
    }, 2000);
  };

  const handleDudo = () => handleChallenge('dudo');
  const handleCalza = () => handleChallenge('calza');

  const resolveChallenge = () => {
    setGameState(prevGameState => {
        if (!prevGameState || !prevGameState.currentBid || prevGameState.lastBidderIndex === null || !prevGameState.challengeType) return prevGameState;

        const { players, currentBid, lastBidderIndex, currentPlayerIndex, challengeType, isSpecialRound, roundBidHistory } = prevGameState;
        let newPlayerAnalysis = { ...prevGameState.playerAnalysis };
        const bidFace = currentBid.face;
        const bidQuantity = currentBid.quantity;

        const allDice = players.flatMap(p => p.dice);
        const actualDiceCount = allDice.filter(d => {
            if (isSpecialRound || bidFace === 1) {
                return d === bidFace;
            } else {
                return d === bidFace || d === 1;
            }
        }).length;

        const challenger = players[currentPlayerIndex];
        const bidder = players[lastBidderIndex];
        
        const wasBidderBluff = actualDiceCount < bidQuantity;

        // --- Update Player Analysis Data ---
        if (roundBidHistory.length > 0) {
            const firstBidData = roundBidHistory[0];
            const firstBidderId = firstBidData.bidderId;
            const firstBid = firstBidData.bid;
            
            const firstBidder = players.find(p => p.id === firstBidderId);

            if (firstBidder) {
                // NEW: A "first bid bluff" is defined as bidding on a face the player has NONE of, AND they have NO wilds (1s).
                // If a player has a wild, their first bid is never considered a bluff.
                const analysis = newPlayerAnalysis[firstBidderId];
                analysis.firstBidBluffs.total++;
                
                let isConsideredBluff = false;
                
                if (isSpecialRound) {
                    // In special rounds, 1s are not wild. Bluff is simply not having the face.
                    const hasFace = firstBidder.dice.some(d => d === firstBid.face);
                    if (!hasFace) {
                        isConsideredBluff = true;
                    }
                } else {
                    // In normal rounds, new rule applies.
                    const hasWilds = firstBidder.dice.some(d => d === 1);
                    if (!hasWilds) {
                        // Only if they have NO wilds, we check if they have the face.
                        const hasFace = firstBidder.dice.some(d => d === firstBid.face);
                        if (!hasFace) {
                            isConsideredBluff = true;
                        }
                    }
                    // If hasWilds is true, isConsideredBluff remains false.
                }

                if (isConsideredBluff) {
                    analysis.firstBidBluffs.count++;
                }
            }
        }

        const bidderAnalysis = newPlayerAnalysis[bidder.id];
        bidderAnalysis.preRevealTendency.total++;
        const bidderLastBidFace = currentBid.face.toString();
        bidderAnalysis.faceBluffPatterns[bidderLastBidFace].totalBids++;

        if (wasBidderBluff) {
            bidderAnalysis.preRevealTendency.bluffCount++;
            bidderAnalysis.faceBluffPatterns[bidderLastBidFace].bluffCount++;
        } else {
            bidderAnalysis.preRevealTendency.strongHandCount++;
        }
        
        // --- Determine Round Outcome ---
        let outcomeDescription = '';
        let modalTitle = '';
        let playerToUpdate: { id: string; diceChange: number } | null = null;
        let roundStarterIndex = -1;

        if (challengeType === 'dudo') {
            modalTitle = `${challenger.name} не верит в ставку!`;
            if (wasBidderBluff) { // Challenger was right
                outcomeDescription = `<strong>${challenger.name}</strong> был прав, ставка не сыграла. <strong>${bidder.name}</strong> теряет кость.`;
                playerToUpdate = { id: bidder.id, diceChange: -1 };
                roundStarterIndex = lastBidderIndex;
            } else { // Bidder was right
                outcomeDescription = `<strong>${bidder.name}</strong> был прав, ставка сыграла. <strong>${challenger.name}</strong> теряет кость.`;
                playerToUpdate = { id: challenger.id, diceChange: -1 };
                roundStarterIndex = currentPlayerIndex;
            }
        } else { // 'calza'
            modalTitle = `${challenger.name} считает, что ставка точная!`;
            if (actualDiceCount === bidQuantity) { // Challenger was right
                const canGainDie = challenger.dice.length < INITIAL_DICE_COUNT;
                if (canGainDie) {
                    outcomeDescription = `Точное попадание! <strong>${challenger.name}</strong> угадал и возвращает кость!`;
                    playerToUpdate = { id: challenger.id, diceChange: 1 };
                } else {
                    outcomeDescription = `Точное попадание! Но у <strong>${challenger.name}</strong> уже максимум костей.`;
                }
                roundStarterIndex = currentPlayerIndex;
            } else { // Challenger was wrong
                outcomeDescription = `Промах! Ставка не была точной. <strong>${challenger.name}</strong> теряет кость.`;
                playerToUpdate = { id: challenger.id, diceChange: -1 };
                roundStarterIndex = currentPlayerIndex;
            }
        }
        
        // --- Construct Log & Modal Messages ---
        const messageForLog = `
          <div class="text-center p-3 my-2 bg-gray-100 text-gray-800 rounded-lg shadow-md">
            <p class="text-xl font-bold text-gray-900 border-b border-gray-300 pb-2 mb-2">Подсчет костей</p>
            <p><strong>${challenger.name}</strong> объявляет <strong>"${challengeType === 'dudo' ? 'Не верю!' : 'Верю!'}"</strong></p>
            <p>На ставку: <strong>${formatBid(currentBid)}</strong> (от <strong>${bidder.name}</strong>)</p>
            <p>На столе: <strong class="text-3xl text-blue-600">${actualDiceCount}</strong></p>
            <p class="mt-2 text-base">${outcomeDescription}</p>
          </div>
        `.trim().replace(/\s+/g, ' ');

        const messageForModal = `Ставка была <strong>${formatBid(currentBid)}</strong>. <br/> На самом деле их <strong>${actualDiceCount}</strong>. <br/><br/> ${outcomeDescription}`;

        setModalContent({ title: modalTitle, body: <p dangerouslySetInnerHTML={{ __html: messageForModal }} /> });

        return {
          ...prevGameState,
          gamePhase: 'round_over',
          roundStarterIndex,
          roundOutcome: { playerToUpdate, actualDiceCount },
          playerAnalysis: newPlayerAnalysis,
          gameLog: [...prevGameState.gameLog, { message: messageForLog }]
        };
    });
  };

  const startNewRound = () => {
    clearBotTimeout(); // Fix: Prevent lingering bot actions from previous rounds.
    setModalContent(null);
    setIsLoading(false);
    
    setGameState(prev => {
        if (!prev) return null;

        const { roundOutcome } = prev;
        let playersWithUpdatedCounts = [...prev.players];

        if (roundOutcome && roundOutcome.playerToUpdate) {
            playersWithUpdatedCounts = prev.players.map(p => {
                if (p.id === roundOutcome.playerToUpdate!.id) {
                    const newDiceCount = p.dice.length + roundOutcome.playerToUpdate!.diceChange;
                    return { ...p, dice: Array(Math.max(0, Math.min(INITIAL_DICE_COUNT, newDiceCount))).fill(1) };
                }
                return p;
            });
            
            // Отслеживаем победителя раунда
            if (roundOutcome.playerToUpdate.diceChange > 0) {
                const roundWinner = playersWithUpdatedCounts.find(p => p.id === roundOutcome.playerToUpdate!.id);
                if (roundWinner) {
                    trackEvent('round_winner', {
                        winner_name: roundWinner.name,
                        winner_type: roundWinner.isHuman ? 'human' : 'bot',
                        is_special_round: prev.isSpecialRound ? 'yes' : 'no',
                        round_number: prev.roundNumber
                    });
                }
            }
        }
        
        const activePlayers = playersWithUpdatedCounts.filter(p => p.dice.length > 0);
        if (activePlayers.length <= 1) {
            const winner = activePlayers[0];
            const newGamesPlayed = gamesPlayed + 1;
            setGamesPlayed(newGamesPlayed);
            
            // Отслеживаем победителя игры
            trackEvent('game_finished', {
                winner_name: winner?.name || 'Unknown',
                winner_type: winner?.isHuman ? 'human' : 'bot',
                total_rounds: prev.roundNumber,
                games_in_session: newGamesPlayed
            });
            
            setModalContent({ title: "Игра окончена!", body: `${winner?.name || 'Кто-то'} остается последним и выигрывает игру!` });
             return {
                ...prev,
                players: playersWithUpdatedCounts,
                gamePhase: 'game_over',
                roundOutcome: null,
                isSpecialRound: false,
                roundBidHistory: [],
                currentBid: null,
                lastBidderIndex: null,
            };
        }

        const newStarterIndex = prev.roundStarterIndex >= 0 ? prev.roundStarterIndex : 0;

        // --- Special Round Check ---
        const potentialSpecialRoundPlayer = playersWithUpdatedCounts.find(p => 
            p.id === playersWithUpdatedCounts[newStarterIndex].id &&
            p.dice.length === 1 && 
            !p.hasUsedSpecialRound &&
            activePlayers.length > 2
        );

        if (potentialSpecialRoundPlayer) {
             if (potentialSpecialRoundPlayer.isHuman) {
                // Let the human decide
                return { ...prev, players: playersWithUpdatedCounts, roundStarterIndex: newStarterIndex, gamePhase: 'special_round_decision' };
             } else {
                // Bot decides based on personality and game state
                if (shouldBotStartSpecialRound(potentialSpecialRoundPlayer, { ...prev, players: playersWithUpdatedCounts })) {
                    addToLog(`объявляет Special раунд!`, potentialSpecialRoundPlayer.id);
                    return proceedWithNewRound(playersWithUpdatedCounts, newStarterIndex, true, potentialSpecialRoundPlayer.id, prev);
                }
             }
        }
        
        return proceedWithNewRound(playersWithUpdatedCounts, newStarterIndex, false, null, prev);
    });
  };
  
  // Helper to finalize starting a new round state
  const proceedWithNewRound = (
    updatedPlayers: Player[], 
    starterIndex: number, 
    isSpecial: boolean,
    specialRoundCallerId: string | null,
    prevState: GameState
    ): GameState => {
        
        let newPlayers = updatedPlayers.map(p => ({
            ...p,
            dice: rollDice(p.dice.length),
            hasUsedSpecialRound: p.hasUsedSpecialRound || p.id === specialRoundCallerId
        }));
        
        let validStarterIndex = starterIndex;
        if (validStarterIndex < 0 || validStarterIndex >= newPlayers.length) {
          validStarterIndex = 0; // Fallback
        }

        // Ensure the starter for the new round is an active player.
        // This loop will find the next player in order who still has dice.
        let guard = 0;
        while (newPlayers[validStarterIndex].dice.length === 0 && guard < newPlayers.length) {
          validStarterIndex = (validStarterIndex + 1) % newPlayers.length;
          guard++;
        }

        const newRoundNumber = prevState.roundNumber + 1;

        return {
            ...prevState,
            players: newPlayers,
            currentPlayerIndex: validStarterIndex,
            roundStarterIndex: validStarterIndex,
            currentBid: null,
            lastBidderIndex: null,
            gamePhase: 'bidding',
            challengeType: null,
            roundBidHistory: [],
            roundOutcome: null,
            isSpecialRound: isSpecial,
            gameLog: [...(prevState.gameLog || []), { message: isSpecial ? `--- Special раунд #${newRoundNumber} ---` : `--- Новый раунд #${newRoundNumber} ---` }],
            roundNumber: newRoundNumber,
        };
  }
  
  const handleSpecialRoundDecision = (startSpecial: boolean) => {
    setGameState(prev => {
        if (!prev) return prev;

        // Use the already updated player counts from the previous state transition.
        const playersWithUpdatedCounts = prev.players; 
        
        const humanPlayerId = playersWithUpdatedCounts.find(p => p.isHuman)?.id || '';

        if (startSpecial) {
             addToLog(`объявляет Special раунд!`, humanPlayerId);
        }
        
        return proceedWithNewRound(
            playersWithUpdatedCounts, 
            prev.roundStarterIndex, 
            startSpecial, 
            startSpecial ? humanPlayerId : null,
            prev
        );
    });
  }

  // Effect for bot turns
  useEffect(() => {
    // If it's not the bidding phase, do nothing.
    if (!gameState || gameState.gamePhase !== 'bidding') return;

    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (!currentPlayer.isHuman) {
      // A bot's turn. If we are already loading, it means a timer is already set, so we do nothing to prevent conflicts.
      if (isLoading) return;

      setIsLoading(true);
      const delay = Math.random() * 1280 + 960;
      botTurnTimeoutRef.current = window.setTimeout(() => {
        const decision = getBotDecision(gameState, currentPlayer);
        
        // Fix: Centralized action handling. All logging is now done by the `handle...` functions
        // to ensure consistency and prevent duplicate log entries.
        switch (decision.decision) {
          case 'DUDO':
            if (gameState.currentBid) {
              handleDudo();
            } else {
              // Fallback for an impossible action
              handlePlaceBid({ quantity: 1, face: 2 });
            }
            break;
          case 'CALZA':
            if (gameState.currentBid) {
              handleCalza();
            } else {
              // Fallback for an impossible action
              handlePlaceBid({ quantity: 1, face: 2 });
            }
            break;
          case 'BID':
            if (decision.bid) {
              // Reuse handlePlaceBid for bots to ensure consistent logging and state updates.
              handlePlaceBid(decision.bid);
            } else {
              // Fallback if bot fails to produce a bid
              if (gameState.currentBid) {
                handleDudo();
              } else {
                handlePlaceBid({ quantity: 1, face: 2 });
              }
            }
            break;
          default:
            // Should not happen, but as a fallback, advance the turn to prevent a hang.
            nextTurn();
        }
      }, delay);
    }
    
    // Cleanup function: this will be called when the component unmounts
    // or when the dependencies (gameState) change.
    return () => {
      clearBotTimeout();
    };
  }, [gameState]); // IMPORTANT: isLoading is removed from dependencies to prevent a race condition.

  // Effect for auto-scrolling
  useEffect(() => {
    if (!gameState) return;

    const timeoutId = setTimeout(() => {
      const currentPlayer = gameState.players[gameState.currentPlayerIndex];
      if (!currentPlayer) return;

      if (currentPlayer.isHuman) {
        bidControlsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else {
        const playerElement = playerRefs.current[currentPlayer.id];
        playerElement?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 700); // Small delay to allow render

    return () => clearTimeout(timeoutId);
  }, [gameState?.currentPlayerIndex, gameState?.roundNumber]);

  if (!gameState) {
    return (
      <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-orange-400 mb-4">Secret Tea Society 🍵</h1>
          <div className="flex items-center justify-center space-x-3">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300"></div>
            <span className="text-2xl font-semibold text-gray-300">Загрузка игры...</span>
          </div>
        </div>
      </div>
    );
  }
  
  const { players, currentPlayerIndex, gamePhase, currentBid, gameLog, lastBidderIndex, roundBidHistory, isSpecialRound } = gameState;
  const totalDiceInPlay = players.reduce((sum, p) => sum + p.dice.length, 0);

  const humanPlayerIndex = players.findIndex(p => p.isHuman);
  let displayPlayers: Player[] = [];
  if (humanPlayerIndex !== -1) {
    const playersInTurnOrder = [
        ...players.slice(humanPlayerIndex),
        ...players.slice(0, humanPlayerIndex)
    ];
    const human = playersInTurnOrder.shift();
    if(human) {
      displayPlayers = [...playersInTurnOrder, human];
    } else {
      displayPlayers = players; // Fallback
    }
  } else {
    displayPlayers = players; // Fallback for no human player
  }
  
  return (
    <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex flex-col items-center font-sans relative overflow-hidden">
      {isSpecialRound && <BackgroundStars />}

      <div className="w-full max-w-7xl text-center mb-6 z-10 relative">
        <h1 className="text-5xl font-bold text-orange-400 mb-2">Secret Tea Society 🍵</h1>
        <p className="text-gray-400 text-lg">Последний игрок с костями побеждает!</p>
        <div className="absolute top-0 right-0">
            <button onClick={() => setIsLogModalOpen(true)} className="p-3 bg-gray-700 rounded-full">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m4 4H3m4 4h12m-4 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>
        </div>
      </div>
      
      <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 lg:gap-8 lg:items-end z-10">
        
        {/* Player List Column */}
        <div className="w-full flex flex-col justify-start space-y-4">
          {displayPlayers.map(p => (
            <PlayerComponent
              ref={el => { playerRefs.current[p.id] = el; }}
              key={p.id} 
              player={p}
              isCurrent={players[currentPlayerIndex].id === p.id}
              gamePhase={gamePhase}
              lastBid={currentBid}
              isLastBidder={players[lastBidderIndex ?? -1]?.id === p.id}
              roundBidHistory={roundBidHistory}
              isSpecialRound={isSpecialRound}
            />
          ))}
        </div>

        {/* Controls Column */}
        <div className="w-full space-y-4">
          {isLoading && (
              <div className="bg-gray-700/50 p-4 rounded-lg flex items-center justify-center space-x-3 w-full max-w-lg mx-auto">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300"></div>
                  <span className="text-xl font-semibold text-yellow-300">{players[currentPlayerIndex].name} думает...</span>
              </div>
          )}
          <div ref={bidControlsRef} className="w-full flex justify-center">
            {players[currentPlayerIndex].isHuman && gamePhase === 'bidding' && (
              <BidControls 
                currentBid={currentBid}
                onPlaceBid={handlePlaceBid}
                onDudo={handleDudo}
                onCalza={handleCalza}
                totalDiceInPlay={totalDiceInPlay}
                isDisabled={isLoading || !players[currentPlayerIndex].isHuman}
                isSpecialRound={isSpecialRound}
                playerDiceCount={players[currentPlayerIndex].dice.length}
              />
            )}
          </div>
        </div>
      </div>

      <Modal
        isOpen={!!modalContent}
        title={modalContent?.title || ''}
        onClose={() => {
          if (gamePhase === 'game_over') {
            const randomNumPlayers = Math.floor(Math.random() * 3) + 3; // 3, 4, or 5
            setupGame(randomNumPlayers);
            setModalContent(null);
          } else {
            startNewRound();
          }
        }}
      >
        {modalContent?.body}
      </Modal>

      {gamePhase === 'special_round_decision' && (
         <SpecialRoundDecisionModal onDecision={handleSpecialRoundDecision} />
      )}

      <LogModal 
        isOpen={isLogModalOpen} 
        onClose={() => setIsLogModalOpen(false)}
        logs={gameLog}
        players={players}
      />

    </div>
  );
};