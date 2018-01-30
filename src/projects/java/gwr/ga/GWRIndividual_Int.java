package gwr.ga;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class GWRIndividual_Int extends GWRIndividual {

	public GWRIndividual_Int(List<Double> chromosome, double sd, double maxGene, double minGene) {
		super(chromosome, sd, maxGene, minGene);
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
		List<Double> nChromosome = new ArrayList<>();
		for (int j = 0; j < chromosome.size(); j++) {
			double h = chromosome.get(j);
			if (r.nextDouble() < 1.0 / chromosome.size()) {
				if (mutationMode) {
					double mean = 0;
					for (int i : cmI.get(j))
						mean += chromosome.get(i);
					mean /= cmI.get(j).size();
					h = getPoissonRandom(mean * sd);
				} else {
					// not weighted yet
					DescriptiveStatistics ds = new DescriptiveStatistics();
					for ( int i : cmI.get(j) )
						ds.addValue( chromosome.get(i) );
					double pc25 = ds.getPercentile(25);
					double pc75 = ds.getPercentile(75);
					double iqr = pc75 - pc25;
					double lower = pc25 - sd*iqr;
					double upper = pc75 + sd*iqr;
					h = (int) Math.round( lower + r.nextDouble() * (upper-lower) );		
				}
				h = Math.max(minGene, Math.min(maxGene, h));
			}
			nChromosome.add(h);
		}
		return new GWRIndividual_Int(nChromosome, sd, maxGene, minGene);
	}

	@Override
	public GWRIndividual recombine(GWRIndividual mother) {
		List<Double> mChromosome = ((GWRIndividual)mother).getChromosome();
		List<Double> nChromosome = new ArrayList<>();
		
		for( int i = 0; i < chromosome.size(); i++)
			if( !meanRecomb ) {
				if( r.nextBoolean() ) 
					nChromosome.add( mChromosome.get(i) );
				else
					nChromosome.add(chromosome.get(i));
			} else {
				double h = Math.round( (chromosome.get(i)+mChromosome.get(i))/2 );
				nChromosome.add( h );
			}
		return new GWRIndividual_Int( nChromosome, sd, maxGene, minGene );
	}
}
