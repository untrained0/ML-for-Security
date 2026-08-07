import MathEq from '@/components/MathEq';

export const EXPLAINERS: Record<string, { title: string; content: React.ReactNode }> = {
  // SVM Parameters
  svm_c: {
    title: 'SVM Parameter C (Regularization)',
    content: (
      <div className="flex flex-col gap-3">
        <p>The <strong className="text-blue-600">C parameter</strong> controls the trade-off between achieving a low training error and a low testing error that is the ability to generalize your classifier to unseen data.</p>
        <div className="bg-gray-50 p-3 rounded-md border border-gray-100 font-mono text-sm">
          <MathEq math="\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i" />
        </div>
        <ul className="list-disc pl-4 space-y-1">
          <li><strong>Large C</strong>: Strict margin. The model tries to classify all training examples correctly, which might lead to <em>overfitting</em> (a jagged boundary).</li>
          <li><strong>Small C</strong>: Soft margin. The model allows some misclassifications for a smoother, more generalized decision boundary.</li>
        </ul>
      </div>
    )
  },
  kernel_gamma: {
    title: 'RBF Kernel Gamma (γ)',
    content: (
      <div className="flex flex-col gap-3">
        <p>The <strong className="text-blue-600">Gamma (γ)</strong> parameter defines how far the influence of a single training example reaches.</p>
        <div className="bg-gray-50 p-3 rounded-md border border-gray-100 font-mono text-sm">
          <MathEq math="K(x, x') = \exp(-\gamma \|x - x'\|^2)" />
        </div>
        <ul className="list-disc pl-4 space-y-1">
          <li><strong>High Gamma</strong>: The reach is small. The model boundary tightly wraps around individual data points (high variance, prone to overfitting).</li>
          <li><strong>Low Gamma</strong>: The reach is large. The model considers points farther away as similar, leading to a smoother boundary.</li>
        </ul>
      </div>
    )
  },
  
  // Attack Parameters
  attack_eta: {
    title: 'Attack Learning Rate (η)',
    content: (
      <div className="flex flex-col gap-3">
        <p>The <strong className="text-blue-600">Learning Rate (η)</strong> controls the step size of the gradient ascent during the poisoning attack.</p>
        <div className="bg-gray-50 p-3 rounded-md border border-gray-100 font-mono text-sm">
          <MathEq math="x_c \leftarrow x_c + \eta \nabla_{x_c} L_{val}(x_c)" />
        </div>
        <p>When computing the gradient of the validation loss with respect to the poison points, the learning rate determines how far the points are moved in the direction of the gradient at each iteration.</p>
        <p className="text-gray-500 text-xs mt-1">If η is too large, the points might jump erratically; if too small, the attack takes too many iterations.</p>
      </div>
    )
  },
  attack_beta: {
    title: 'Attack Regularization (β)',
    content: (
      <div className="flex flex-col gap-3">
        <p>The <strong className="text-blue-600">Attack Regularization (β)</strong> ensures the poison points don't drift too far from the data distribution.</p>
        <div className="bg-gray-50 p-3 rounded-md border border-gray-100 font-mono text-sm">
          <MathEq math="L_{attack} = L_{val}(D_{val}) - \beta \sum_{x \in D_p} \|x - \text{center}\|^2" />
        </div>
        <p>The attacker adds a penalty (weighted by β) to the objective function, pulling the poison points toward the target class cluster. Without this, the points might drift into empty space where they have less influence!</p>
      </div>
    )
  },

  // Regression Parameters
  ridge_lambda: {
    title: 'Regression Regularization (λ)',
    content: (
      <div className="flex flex-col gap-3">
        <p>The <strong className="text-blue-600">Regularization parameter (λ)</strong> determines the strength of the penalty applied to the model weights to prevent overfitting.</p>
        <ul className="list-disc pl-4 space-y-1">
          <li><strong>OLS (λ=0)</strong>: No penalty. Fits the data exactly but is highly susceptible to outliers.</li>
          <li><strong>Ridge (L2)</strong>: Penalizes <MathEq math="\lambda \|w\|_2^2" />. Shrinks weights toward zero.</li>
          <li><strong>LASSO (L1)</strong>: Penalizes <MathEq math="\lambda \|w\|_1" />. Can drive certain weights to exactly zero, performing feature selection!</li>
        </ul>
      </div>
    )
  },

  // Math Panel Variables
  theta_weights: {
    title: 'Model Weights (θ)',
    content: (
      <div className="flex flex-col gap-3">
        <p>The <strong className="text-blue-600">Weight Vector (θ)</strong> contains the coefficients of the regression line. For a 1D dataset, it includes the slope $w$ and the bias $b$.</p>
        <p>For Ridge Regression, the optimal weights are found using a closed-form algebraic solution:</p>
        <div className="bg-gray-50 p-3 rounded-md border border-gray-100 font-mono text-sm flex justify-center">
          <MathEq math="\theta = (X^T X + \lambda I)^{-1} X^T y" />
        </div>
        <p className="text-gray-500 text-xs">The attack works by modifying the matrix $X$ and vector $y$ to skew these weights!</p>
      </div>
    )
  },
  
  mse_metric: {
    title: 'Mean Squared Error (MSE)',
    content: (
      <div className="flex flex-col gap-3">
        <p>The <strong className="text-blue-600">Mean Squared Error</strong> is the average of the squared differences between the predicted values (${"\\hat{y}"}$) and actual values ($y$).</p>
        <div className="bg-gray-50 p-3 rounded-md border border-gray-100 font-mono text-sm flex justify-center">
          <MathEq math="\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2" />
        </div>
        <p>The attacker's ultimate goal in this simulator is to <strong>maximize</strong> the validation MSE, forcing the model to make terrible predictions on clean, unseen data.</p>
      </div>
    )
  },

  kkt_gradient: {
    title: 'KKT Implicit Differentiation',
    content: (
      <div className="flex flex-col gap-3">
        <p>How does the attacker compute the gradient of the validation loss with respect to their poison points? They use <strong>Implicit Differentiation</strong> on the Karush-Kuhn-Tucker (KKT) optimality conditions.</p>
        <div className="bg-gray-50 p-3 rounded-md border border-gray-100 font-mono text-[11px] overflow-x-auto">
          <MathEq math="\nabla_{x_c} W = - (\nabla_{ww}^2 L_{tr})^{-1} \nabla_{w x_c}^2 L_{tr}" />
        </div>
        <p>This beautiful math trick allows the attacker to differentiate "through" the training process itself, finding exactly how moving a poison point $x_c$ changes the final weights $W$!</p>
      </div>
    )
  }
};
