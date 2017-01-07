public class GaussianNaiveBayesClassifier {

	boolean fitted;
	int[] classes;


	public GaussianNaiveBayesClassifier() {

		fitted = false;

	}

	public void fit(double[][] X, double[] y) {
		// Fit the model
  		//
  		// Note: Calling `fit()` overwrites previous information. Use `partial_fit()`
 		//  to update the model with new training data.


		fitted = false; // if model has already been trained, re-initialize parameters

		partial_fit(X,y);
	}

	public void partialFit(double[][] X, double[] y) {
		// Fit or update the model
		//
		// Using `partialFit()` allows to update the model given new data.

		if (!fitted) { // model has not been trained yet, initialize parameters
			// ...
		}

		// Update class prior
		// ...

		// Update sum and mean
		// ...

		// Update sum of squares and variance
		// ...

		fitted = true;

	}

	public int[] predict(double[][] X) {
		// ...
	}

	public float[] predictProba(double[][] X) {
		// ...
	}

	public float score(double[][] X, double[] y) {
		// ...
	}

	public float[][] getMeans() {
		// ...
	}

	public float[][] getVariances() {
		// ...
	}

	public float[][] getClassPriors() {
		// ...
	}

	public void setMeans() {
		// ...
	}

	public void setVariances() {
		// ...
	}

	public void setClassPriors() {
		// ...
	}

	public double[][] decisionBoundary() {
		// ...
	}

	private double[][] gaussian(double[][] X, double[] mu, double[] var) {
		// ...
	}

	public static void main(String[] args) {

		GaussianNaiveBayesClassifier clf = new GaussianNaiveBayesClassifier();

		// ...

	}
}