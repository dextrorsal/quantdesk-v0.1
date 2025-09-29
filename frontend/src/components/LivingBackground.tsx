import React, { useState, useEffect } from 'react'

interface TradingScript {
  id: string
  name: string
  color: string
  content: string[]
  duration: number
}

const LivingBackground: React.FC = () => {
  const [currentScript, setCurrentScript] = useState<TradingScript | null>(null)
  const [scriptIndex, setScriptIndex] = useState(0)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [displayedText, setDisplayedText] = useState<string>('')
  const [currentLineIndex, setCurrentLineIndex] = useState(0)
  const [isTyping, setIsTyping] = useState(false)

  const tradingScripts: TradingScript[] = [
    {
      id: 'signal_generation',
      name: 'Signal Generation Engine',
      color: 'green',
      duration: 4000,
      content: [
        'signal_gen --analyze --market --depth=1000',
        'scanning market patterns...',
        'analyzing price action...',
        'detecting momentum shifts...',
        'calculating support/resistance...',
        'generating trading signals...',
        'validating signal accuracy...',
        'optimizing parameters...',
        'updating model weights...',
        'testing signal performance...',
        '✓ Generated 12 new signals',
        '✓ Accuracy: 87.3%',
        '✓ Confidence: High',
        'signal_gen --complete'
      ]
    },
    {
      id: 'portfolio_management',
      name: 'Portfolio Management System',
      color: 'blue',
      duration: 4500,
      content: [
        'portfolio_mgr --rebalance --safety --auto',
        'monitoring portfolio health...',
        'calculating risk metrics...',
        'analyzing position sizes...',
        'checking margin requirements...',
        'rotating profits to safety...',
        'rebalancing allocations...',
        'optimizing returns...',
        'updating risk parameters...',
        'validating strategies...',
        '✓ Rotated $2.4K to safety pot',
        '✓ Portfolio value: $45.2K',
        '✓ Risk level: Optimal',
        'portfolio_mgr --complete'
      ]
    },
    {
      id: 'backtesting_engine',
      name: 'Backtesting Engine',
      color: 'orange',
      duration: 5000,
      content: [
        'backtest --strategy --momentum --historical=2y',
        'loading historical data...',
        'initializing test environment...',
        'running strategy simulation...',
        'testing momentum signals...',
        'validating algorithm logic...',
        'optimizing parameters...',
        'analyzing performance metrics...',
        'calculating risk metrics...',
        'generating reports...',
        '✓ Tested momentum strategy',
        '✓ Success rate: 92.1%',
        '✓ Sharpe ratio: 1.84',
        'backtest --complete'
      ]
    },
    {
      id: 'ml_training',
      name: 'ML Model Training',
      color: 'purple',
      duration: 5500,
      content: [
        'ml_train --lstm --optimize --epochs=100',
        'initializing neural network...',
        'loading training data...',
        'preprocessing features...',
        'training LSTM model...',
        'optimizing hyperparameters...',
        'validating model performance...',
        'testing on unseen data...',
        'updating model weights...',
        'saving trained model...',
        '✓ Trained LSTM model',
        '✓ Accuracy: 89.7%',
        '✓ Loss: 0.0234',
        'ml_train --complete'
      ]
    },
    {
      id: 'risk_management',
      name: 'Risk Management System',
      color: 'red',
      duration: 3500,
      content: [
        'risk_mgr --monitor --real-time --alerts',
        'scanning market volatility...',
        'calculating VaR metrics...',
        'monitoring position exposure...',
        'checking correlation risks...',
        'analyzing drawdown limits...',
        'updating risk parameters...',
        'generating risk reports...',
        'setting stop losses...',
        '✓ Risk level: Low',
        '✓ VaR: $1.2K',
        '✓ Max drawdown: 3.2%',
        'risk_mgr --complete'
      ]
    },
    {
      id: 'order_execution',
      name: 'Order Execution Engine',
      color: 'cyan',
      duration: 3000,
      content: [
        'order_exec --smart --routing --optimize',
        'analyzing market depth...',
        'calculating optimal routing...',
        'executing smart orders...',
        'minimizing slippage...',
        'updating order book...',
        'confirming executions...',
        '✓ Executed 5 orders',
        '✓ Avg slippage: 0.02%',
        '✓ Fill rate: 98.7%',
        'order_exec --complete'
      ]
    }
  ]

  // Typing effect
  useEffect(() => {
    if (!currentScript || isTransitioning) return

    const typeText = () => {
      setIsTyping(true)
      setDisplayedText('')
      setCurrentLineIndex(0)
      
      let lineIndex = 0
      let charIndex = 0
      let fullText = ''
      
      const typeInterval = setInterval(() => {
        if (lineIndex < currentScript.content.length) {
          const currentLine = currentScript.content[lineIndex]
          
          if (charIndex < currentLine.length) {
            fullText += currentLine[charIndex]
            setDisplayedText(fullText)
            charIndex++
          } else {
            fullText += '\n'
            setDisplayedText(fullText)
            lineIndex++
            charIndex = 0
          }
        } else {
          setIsTyping(false)
          clearInterval(typeInterval)
        }
      }, 10) // Much faster typing
      
      return () => clearInterval(typeInterval)
    }

    const timeout = setTimeout(typeText, 200)
    return () => clearTimeout(timeout)
  }, [currentScript, isTransitioning])

  useEffect(() => {
    const cycleScripts = () => {
      if (isTransitioning) return

      setIsTransitioning(true)
      
      // Smooth transition - no flashing
      setTimeout(() => {
        setCurrentScript(tradingScripts[scriptIndex])
        setScriptIndex((prev) => (prev + 1) % tradingScripts.length)
        setIsTransitioning(false)
      }, 100)
    }

    // Start with first script
    if (!currentScript) {
      setCurrentScript(tradingScripts[0])
      setScriptIndex(1)
    }

    // Set up cycling - simple timing
    const interval = setInterval(cycleScripts, currentScript?.duration || 3500)

    return () => clearInterval(interval)
  }, [currentScript, scriptIndex, isTransitioning])

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'green':
        return 'text-green-400'
      case 'blue':
        return 'text-blue-400'
      case 'orange':
        return 'text-orange-400'
      case 'purple':
        return 'text-purple-400'
      case 'red':
        return 'text-red-400'
      case 'cyan':
        return 'text-cyan-400'
      default:
        return 'text-gray-400'
    }
  }

  if (!currentScript) return null

  return (
    <div className="absolute top-0 left-0 pointer-events-none z-10 opacity-25 w-full min-h-screen">
      <div className="w-full h-full flex items-start justify-start pt-48 pl-8">
        <div className={`font-mono text-sm ${getColorClasses(currentScript.color)} transition-opacity duration-500 ${isTransitioning ? 'opacity-0' : 'opacity-100'} whitespace-pre-wrap max-w-2xl`}>
          {displayedText}
          {isTyping && <span className="text-gray-400 animate-pulse">_</span>}
        </div>
      </div>
    </div>
  )
}

export default LivingBackground