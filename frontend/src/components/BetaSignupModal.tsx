import React from 'react';
import { X, ArrowRight } from 'lucide-react';

interface BetaSignupModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const BetaSignupModal: React.FC<BetaSignupModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-black border border-purple-500 rounded-lg p-6 max-w-md w-full relative shadow-2xl">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
        
        {/* Popup content */}
        <div className="text-center">
          <div className="mb-4">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-purple-500 bg-opacity-20 border border-purple-500 rounded-full mb-4">
              <ArrowRight className="h-6 w-6 text-purple-400" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              Limited Beta Access
            </h3>
            <p className="text-gray-300 text-sm">
              Be among the first to experience QuantDesk's advanced trading platform. 
              Apply now for exclusive early access.
            </p>
          </div>
          
          <div className="space-y-3">
            <a 
              href="https://docs.google.com/forms/d/e/1FAIpQLSfG--SHQ6hifTo5S9p0LbdmC3mrds0cTjIwvA2CgQ8hoOnCwA/viewform?usp=dialog" 
              target="_blank" 
              rel="noopener noreferrer"
              className="group inline-flex items-center space-x-2 bg-black hover:bg-purple-400 border border-purple-400 hover:border-purple-400 text-purple-400 hover:text-black font-semibold py-3 px-8 rounded-lg transition-all duration-300 hover:shadow-lg hover:shadow-purple-400/50"
            >
              <span>Sign Up</span>
              <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </a>
            <button
              onClick={onClose}
              className="block w-full text-gray-400 hover:text-white text-sm transition-colors"
            >
              Maybe Later
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BetaSignupModal;
