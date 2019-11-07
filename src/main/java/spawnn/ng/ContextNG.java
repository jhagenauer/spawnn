package spawnn.ng;

import java.util.List;

import spawnn.ng.sorter.SorterContext;
import spawnn.ng.sorter.SorterWMC;
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
		double[] context = ((SorterContext)sorter).getContext(x);
		if( sorter instanceof SorterWMC )
			((SorterWMC)sorter).sort(x, neurons, context);
		else
			sorter.sort(x, neurons);
					
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
			
		// adapt
		for (int k = 0; k < neurons.size(); k++) {
			double adapt = e * Math.exp((double) -k / l);	
			double[] w = neurons.get(k);
			
			// adapt weights and adapt context vector part
			for (int i = 0; i < x.length; i++) {
				w[i] += adapt * (x[i] - w[i]);
				w[x.length + i] += adapt * (context[i] - w[x.length + i]);
			}
		}
	}
}
