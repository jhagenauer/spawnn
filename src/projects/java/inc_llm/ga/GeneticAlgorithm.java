package inc_llm.ga;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import regionalization.ga.GAIndividual;

public class GeneticAlgorithm {

	private static Logger log = Logger.getLogger(GeneticAlgorithm.class);
	private final static Random r = new Random();
	
	public int tournamentSize = 2;
	public double recombProb = 0.6;

	public GAIndividual search(List<GAIndividual> init) {
		List<GAIndividual> gen = new ArrayList<GAIndividual>(init);
		GAIndividual best = null;

		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize * 2;

		int maxK = 1000;
		int k = 0;
		while (k < maxK /* || noImpro < 100 */) {

			// check best and increase noImpro
			noImpro++;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (GAIndividual cur : gen) {
				if (best == null || cur.getValue() < best.getValue()) {
					best = cur;
					noImpro = 0;
					log.debug( Arrays.toString(((ParamIndividual)best).getChromosome())+", "+best.getValue());
				}
				ds.addValue(cur.getValue());
			}
			if (noImpro == 0 || k % 10 == 0)
				log.info(k+","+ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );

			// SELECT NEW GEN/POTENTIAL PARENTS
			// elite
			Collections.sort(gen);
			List<GAIndividual> elite = new ArrayList<GAIndividual>();
			// elite.addAll( gen.subList(0, Math.max( 1, (int)( 0.01*gen.size() ) ) ) );
			gen.removeAll(elite);

			List<GAIndividual> selected = new ArrayList<GAIndividual>(elite);
			while (selected.size() < parentSize) {
				GAIndividual i = tournament(gen, tournamentSize);
				selected.add(i);
			}
			gen = selected;

			// GENERATE OFFSPRING
			List<GAIndividual> offspring = new ArrayList<GAIndividual>();

			for (int i = 0; i < offspringSize; i++) {
				final GAIndividual a = gen.get(r.nextInt(gen.size()));
				final GAIndividual b = gen.get(r.nextInt(gen.size()));

				GAIndividual child;

				if (r.nextDouble() < recombProb)
					child = a.recombine(b);
				else
					child = a;

				offspring.add(child.mutate());
			}

			gen.clear();
			gen.addAll(offspring);

			k++;
		}
		log.debug(k);
		return best;
	}

	// tournament selection
	public GAIndividual tournament(List<GAIndividual> gen, int k) {
		List<GAIndividual> ng = new ArrayList<GAIndividual>();

		double sum = 0;
		for (int i = 0; i < k; i++) {
			GAIndividual in = gen.get(r.nextInt(gen.size()));
			ng.add(in);
			sum += in.getValue();
		}

		Collections.sort(ng);

		// deterministic
		return ng.get(0);
	}

	public GAIndividual binaryProbabilisticTournament(List<GAIndividual> gen, double prob) {
		Random r = new Random();
		GAIndividual a = gen.get(r.nextInt(gen.size()));
		GAIndividual b = gen.get(r.nextInt(gen.size()));

		if (b.getValue() < a.getValue()) {
			GAIndividual tmp = a;
			a = b;
			b = tmp;
		}
		if (r.nextDouble() < prob)
			return a;
		else
			return b;
	}

	// roulette wheel selection
	public GAIndividual rouletteWheelSelect(List<GAIndividual> gen) {
		double sum = 0;
		for (GAIndividual in : gen)
			sum += in.getValue();

		Random r = new Random();
		double v = r.nextDouble();

		double a = 0, b = 0;
		for (int j = 0; j < gen.size(); j++) {
			a = b;
			b = (sum - gen.get(j).getValue()) / sum + b;
			if (a <= v && v <= b || j + 1 == gen.size() && a <= v)
				return gen.get(j);
		}
		return null;
	}

	// stochastic universal sampling
	public List<GAIndividual> sus(List<GAIndividual> gen, int n) {
		List<GAIndividual> l = new ArrayList<GAIndividual>();
		Collections.sort(gen);

		double sum = 0;
		for (GAIndividual in : gen)
			sum += in.getValue();

		// intervals
		double ivs[] = new double[gen.size() + 1];
		ivs[0] = 0.0f;
		for (int j = 0; j < ivs.length - 1; j++)
			ivs[j + 1] = sum - gen.get(j).getValue() + ivs[j];

		double start = r.nextDouble() * sum / n;
		for (int i = 0; i < n; i++) {
			double v = start + i * sum / n;
			// binary search of v
			int first = 0, last = ivs.length - 1;
			while (true) {
				int mid = first + (last - first) / 2;

				if (last - first <= 1) {
					l.add(gen.get(mid));
					break;
				}
				if (ivs[first] <= v && v <= ivs[mid])
					last = mid;
				else if (ivs[mid] <= v && v <= ivs[last])
					first = mid;
			}
		}
		return l;
	}

	public static void main(String[] args) {
		ParamIndividual.numVal = 4;
		
		printPI(new int[]{5, 6, 10, 6, 4, 5, 1, 0});
		printPI(new int[]{6, 8, 7, 9, 4, 5, 1, 0});
		
		// problem: its best do chose lambda so that it exactly fits into t_max
		
		List<GAIndividual> init = new ArrayList<GAIndividual>();
		while (init.size() < 50)
			init.add(new ParamIndividual());

		GeneticAlgorithm gen = new GeneticAlgorithm();
		gen.search(init);

	}
	
	public static void printPI( int[] c ) {
		ParamIndividual pi = new ParamIndividual(c);
		log.debug(Arrays.toString(pi.getChromosome())+":"+pi.paramsToString()+":"+pi.getValue() );
	}

}
