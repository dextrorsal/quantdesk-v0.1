import React, { createContext, useContext, ReactNode, useMemo } from 'react';

interface WalletContextType {
  connected: boolean;
  publicKey: string | null;
  wallet: any;
  signMessage: any;
  disconnect: () => void;
}

const WalletContext = createContext<WalletContextType | undefined>(undefined);

interface MockWalletProviderProps {
  children: ReactNode;
}

export const MockWalletProvider: React.FC<MockWalletProviderProps> = ({ children }) => {
  const walletValue = useMemo(() => ({
    connected: false,
    publicKey: null,
    wallet: null,
    signMessage: null,
    disconnect: () => {
      console.log('Mock wallet disconnect called');
    },
  }), []);

  return (
    <WalletContext.Provider value={walletValue}>
      {children}
    </WalletContext.Provider>
  );
};

export const useWallet = () => {
  const context = useContext(WalletContext);
  if (context === undefined) {
    // Return mock values instead of throwing error
    return {
      connected: false,
      publicKey: null,
      wallet: null,
      signMessage: null,
      disconnect: () => {},
    };
  }
  return context;
};
