package myga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

public class GeneticAlgorithm<T extends GAIndividual> {
	
	private static Logger log = Logger.getLogger(GeneticAlgorithm.class);
	
	private final static Random r = new Random();
	
	Evaluator<T> evaluator;
			
	public int tournamentSize = 2;
	public double recombProb = 0.9;
	
	public int evaluations = 0;
				
	public GeneticAlgorithm(Evaluator<T> evaluator) {
		this.evaluator = evaluator;
	}

	public T search( List<T> init ) {		
		List<T> gen = new ArrayList<T>(init);
		T best = null;
		
		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize;
		
		evaluations = init.size();
				
		// init evaluation
		for( T i : init )
			i.setValue(evaluator.evaluate(i));
							
		int k = 0;
		while( noImpro < 400  /*&& evaluations < 400000*/  ) {
						
			// check best and increase noImpro
			noImpro++;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( T cur : gen ) {
				if( best == null || cur.getValue() < best.getValue()  ) { 
					best = cur;
					noImpro = 0;
					log.debug("found best: "+cur.getValue()+", "+cur+", k: "+k);
				}
				ds.addValue( cur.getValue() );
			}
			if( noImpro == 0 || k % 100 == 0 || false ) {		
				log.debug(k+","+noImpro+","+evaluations+","+ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );
			}
															
			// SELECT NEW GEN/POTENTIAL PARENTS
			// elite	
			Collections.sort( gen );	
			List<T> elite = new ArrayList<T>(gen.subList(0, Math.max( 1, (int)( 0.05*gen.size() ) ) ));
			gen.removeAll(elite);																		
			
			List<T> selected = new ArrayList<T>(elite);
			while( selected.size() < parentSize ) {
				// TOURNAMENT SELECTION
				T i = tournament( gen, tournamentSize ); 			
				selected.add( i );
			}		
			gen = selected;	
									
			// GENERATE OFFSPRING
			List<T> offSpring = new ArrayList<T>();
			while( offSpring.size() < offspringSize ) {
				final T a = gen.get( r.nextInt( gen.size() ) );
				final T b = gen.get( r.nextInt( gen.size() ) );
				
				T child;
				
				if( r.nextDouble() < recombProb )
					child = (T)a.recombine( b );
				else 
					child = (T)a.recombine( a ); // clone
																		
				child.mutate();
				child.setValue(evaluator.evaluate( child ));
				offSpring.add(child);
					
			}
			evaluations += offspringSize;
			
			gen = offSpring; // REPLACE GEN WITH OFFSPRING
			//gen.addAll(elite); // ALWAYS KEEP ELITE, is that a good idea?
			k++;
		}			
		return best;
	}
		
	// deterministic tournament selection
	public T tournament( List<T> gen, int k ) {
		List<T> ng = new ArrayList<T>();
		
		for( int i = 0; i < k; i++ ) {
			T in = gen.get( r.nextInt( gen.size() ) );
			ng.add( in );
		}
		
		Collections.sort( ng );
		return ng.get( 0 ); 		
	}
	
	public T binaryProbabilisticTournament( List<T> gen, double prob ) {
		Random r = new Random();
		T a = gen.get( r.nextInt( gen.size() ) );
		T b = gen.get( r.nextInt( gen.size() ) );
		
		if( b.getValue() < a.getValue() ) {
			T tmp = a;
			a = b;
			b = tmp;
		}
		if( r.nextDouble() < prob )
			return a;
		else
			return b;
	}
	
	// roulette wheel selection
	public T rouletteWheelSelect( List<T> gen ) {
		double sum = 0;
		for( T in : gen )
			sum += in.getValue();
				
		Random r = new Random();
		double v = r.nextDouble();
		
		double a = 0, b = 0;
		for( int j = 0; j < gen.size(); j++ ) {
			a = b;
			b = (sum - gen.get(j).getValue())/sum + b;
			if( a <= v && v <= b || j+1 == gen.size() && a <= v ) 
				return gen.get(j);
		}
		return null;
	}

	// stochastic universal sampling
	public List<T> sus( List<T> gen, int n ) {
		List<T> l = new ArrayList<T>();
		Collections.sort( gen );
		
		double sum = 0;
		for( T in : gen )
			sum += in.getValue();

		// intervals
		double ivs[] = new double[gen.size()+1];
		ivs[0] = 0.0f;
		for( int j = 0; j < ivs.length-1; j++ )  
			ivs[j+1] = sum - gen.get(j).getValue() + ivs[j];
		
		double start = r.nextDouble()*sum/n;
		for( int i = 0; i < n; i++ ) {
			double v = start+i*sum/n;
			// binary search of v
			int first = 0, last = ivs.length-1;
			while( true ) {
				int mid = first+(last-first)/2;
				
				if( last - first <= 1 ) {
					l.add( gen.get(mid) );
					break; 
				}
				if( ivs[first] <= v && v <= ivs[mid] ) 
					last = mid;
				else if( ivs[mid] <= v && v <= ivs[last] ) 
					first = mid;
			}
		}
		return l;
	}
}
