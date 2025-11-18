import React, { useEffect, useRef } from 'react';

interface HelpModalProps {
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ onClose }) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);

    // Focus close button when modal opens
    setTimeout(() => {
      closeButtonRef.current?.focus();
    }, 100);

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="help-modal-title"
    >
      <div
        ref={modalRef}
        className="bg-gray-800 rounded-lg shadow-xl p-8 max-w-2xl w-full m-4 max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="help-modal-title" className="text-3xl font-bold text-orange-400 mb-6">
          Что такое Perudo?
        </h2>
        
        <div className="text-gray-300 text-lg space-y-4 mb-6">
          <p>
            <strong className="text-white">Perudo</strong> — это игра в кости с блефом и неполной информацией. 
            Цель игры — остаться последним игроком с костями.
          </p>

          <div>
            <h3 className="text-xl font-semibold text-white mb-2">Основные правила:</h3>
            <ul className="list-disc list-inside space-y-2 ml-2">
              <li>Каждый игрок начинает игру с <strong className="text-white">5 костями</strong></li>
              <li>Игроки по очереди делают <strong className="text-white">ставки</strong> на общее количество костей определенного значения (например, "3 пятерки")</li>
              <li>Каждая следующая ставка должна быть выше предыдущей (больше количество или больше значение)</li>
              <li>Единицы (1) считаются <strong className="text-white">джокерами</strong> и могут быть засчитаны как любое значение</li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold text-white mb-2">Действия игрока:</h3>
            <ul className="list-disc list-inside space-y-2 ml-2">
              <li>
                <strong className="text-white">Ставка (Bid)</strong> — сделать ставку на количество костей определенного значения
              </li>
              <li>
                <strong className="text-white">Оспорить (Challenge)</strong> — оспорить ставку предыдущего игрока. 
                Если ставка неверна, предыдущий игрок теряет кость. Если верна — вы теряете кость.
              </li>
              <li>
                <strong className="text-white">Поверить (Believe)</strong> — все игроки показывают кости, 
                проверяется точное совпадение с последней ставкой
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold text-white mb-2">Особые правила:</h3>
            <ul className="list-disc list-inside space-y-2 ml-2">
              <li>
                <strong className="text-white">Специальный раунд (Palifico)</strong> — активируется автоматически, 
                когда у игрока остается только 1 кость. В этом раунде единицы НЕ являются джокерами.
              </li>
              <li>Игрок, потерявший все кости, выбывает из игры</li>
              <li>Последний игрок с костями становится победителем</li>
            </ul>
          </div>

          <div className="bg-gray-700/50 p-4 rounded-lg mt-4">
            <p className="text-sm">
              Если что-то пошло не так — пиши в ТГ{' '}
              <a
                href="https://t.me/birdlab"
                target="_blank"
                rel="noopener noreferrer"
                className="text-yellow-400 hover:text-yellow-300 underline font-semibold"
              >
                birdlab
              </a>
            </p>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            ref={closeButtonRef}
            onClick={onClose}
            className="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-3 px-8 rounded-lg transition-colors text-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
            aria-label="Закрыть окно помощи"
          >
            Закрыть
          </button>
        </div>
      </div>
    </div>
  );
};

