

import React from 'react';

interface ModalProps {
  isOpen: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, title, onClose, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg shadow-xl p-8 max-w-lg w-full m-4">
        <h2 className="text-3xl font-bold text-white mb-4">{title}</h2>
        <div className="text-gray-300 mb-6 text-lg">{children}</div>
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-3 px-8 rounded-lg transition-colors text-lg"
          >
            Продолжить
          </button>
        </div>
      </div>
    </div>
  );
};

export default Modal;