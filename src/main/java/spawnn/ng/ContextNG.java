package spawnn.ng;

import java.util.List;

import spawnn.ng.sorter.SorterContext;
import spawnn.som.decay.DecayFunction;

public class ContextNG extends NG {
	
	public ContextNG( List<double[]> neurons, DecayFunction neighborhood, DecayFunction adaptation, SorterContext sorter  ) {
		super(neurons,neighborhood,adaptation,sorter);
	}
	
	@Deprecated
	public ContextNG( List<double[]> neurons, double lInit, double lFinal, double eInit, double eFinal, SorterContext sorter  ) {
		super(neurons,lInit,lFinal,eInit,eFinal,sorter);
	}

	@Deprecated
	public ContextNG(int numNeurons, double lInit, double lFinal, double eInit, double eFinal, int dim, SorterContext sorter) {
		super(numNeurons, lInit, lFinal, eInit, eFinal, dim, sorter);
	}
		
	@Override
	public void train(double t, double[] x) {
		double[] context = ((SorterContext)sorter).getContext(x); // sort affects context
		sortNeurons(x);
					
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
			
		// adapt
		for (int k = 0; k < neurons.size(); k++) {
			double adapt = e * Math.exp((double) -k / l);
						
			double[] w = neurons.get(k);
			
			// adapt weights
			for (int i = 0; i < x.length; i++)
				w[i] += adapt * (x[i] - w[i]);
			
			// adapt context vector part
			if( context != null )
				for( int i = 0; i < context.length; i++ )
					w[ x.length + i ] += adapt * (context[i] - w[x.length + i]);
		}
	}
}
