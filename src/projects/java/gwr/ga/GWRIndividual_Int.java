package gwr.ga;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class GWRIndividual_Int extends GWRIndividual {

	public GWRIndividual_Int(List<Double> chromosome, double sd) {
		super(chromosome, sd);
	}

	private int getPoissonRandom(double mean) {
		double L = Math.exp(-mean);
		int k = 0;
		double p = 1.0;
		do {
			p = p * r.nextDouble();
			k++;
		} while (p > L);
		return k - 1;
	}

	@Override
	public GWRIndividual mutate() {
		List<Double> nBw = new ArrayList<>();
		for (int j = 0; j < chromosome.size(); j++) {
			double h = chromosome.get(j);
			if (r.nextDouble() < 1.0 / chromosome.size()) {
				if (mutationMode) {
					double mean = 0;
					for (int i : cmI.get(j))
						mean += chromosome.get(i);
					mean /= cmI.get(j).size();
					h = getPoissonRandom(mean);
				} else {
					// not weighted yet
					DescriptiveStatistics ds = new DescriptiveStatistics();
					for( int i : cmI.get(j) )
						ds.addValue( chromosome.get(i) );
					double pc25 = ds.getPercentile(0.25);
					double pc75 = ds.getPercentile(0.75);
					double iqr = pc75 - pc25;
					h = (int)Math.round( pc25 - sd*iqr + r.nextDouble()*(pc75+sd*iqr) ); 
				}
			}
			nBw.add( h );
		}
		return new GWRIndividual_Int(nBw, sd);
	}

	@Override
	public GWRIndividual recombine(GWRIndividual mother) {
		List<Double> mBw = ((GWRIndividual) mother).getChromosome();
		List<Double> nBw = new ArrayList<>();
		for (int i = 0; i < chromosome.size(); i++)
			if (r.nextBoolean())
				nBw.add(mBw.get(i));
			else
				nBw.add(chromosome.get(i));
		return new GWRIndividual_Int(nBw, sd);
	}
}
