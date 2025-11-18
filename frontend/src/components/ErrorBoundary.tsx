import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-800 text-white flex items-center justify-center p-8">
          <div className="max-w-2xl w-full bg-gray-900 rounded-lg p-8">
            <h1 className="text-3xl font-bold text-red-400 mb-4">Something went wrong</h1>
            <p className="text-gray-300 mb-4">
              An unexpected error occurred. Please refresh the page or try again later.
            </p>
            {this.state.error && (
              <details className="mt-4">
                <summary className="cursor-pointer text-gray-400 hover:text-gray-300 mb-2">
                  Error details
                </summary>
                <pre className="bg-gray-800 p-4 rounded text-sm text-red-300 overflow-auto">
                  {this.state.error.toString()}
                  {this.state.error.stack && (
                    <div className="mt-2 text-xs text-gray-500">
                      {this.state.error.stack}
                    </div>
                  )}
                </pre>
              </details>
            )}
            <button
              onClick={() => window.location.reload()}
              className="mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}


