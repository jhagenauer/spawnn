package spawnn.ng;

import java.util.Collection;

import spawnn.ng.sorter.SorterContext;

public class ContextNG extends NG {
	
	public ContextNG( Collection<double[]> neurons, double lInit, double lFinal, double eInit, double eFinal, SorterContext bg  ) {
		super(neurons,lInit,lFinal,eInit,eFinal,bg);
	}

	public ContextNG(int numNeurons, double lInit, double lFinal, double eInit, double eFinal, int dim, SorterContext bg) {
		super(numNeurons, lInit, lFinal, eInit, eFinal, dim, bg);
	}
		
	@Override
	public double[] train(double t, double[] x) {
		sortNeurons(x);
				
		double[] context = ((SorterContext)sorter).getContext(x);
				
		double l = adaptionRate.getValue(t);
		double e = stepSizeRate.getValue(t);
			
		// adapt
		for (int k = 0; k < neurons.size(); k++) {
			double adapt = e * Math.exp((double) -k / l);
						
			double[] w = neurons.get(k);
			
			// adapt weights
			for (int i = 0; i < x.length; i++)
				w[i] += adapt * (x[i] - w[i]);
			
			// adapt context
			if( context != null )
				for( int i = 0; i < context.length; i++ )
					w[ x.length + i ] += adapt * (context[i] - w[x.length + i]);
		}
		return neurons.get(0);
	}
}
