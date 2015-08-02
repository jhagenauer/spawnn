package moomap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.singleObjective.geneticAlgorithm.gGA;
import jmetal.operators.selection.BinaryTournament;
import jmetal.util.JMException;

import moomap.jmetal.encodings.variable.TopoMap;
import moomap.jmetal.operators.crossover.TopoMapNBCrossover;
import moomap.jmetal.operators.mutation.RndVectorMutation;
import moomap.jmetal.problems.TopoMapSingle;
import moomap.myga.GAIndividual;
import moomap.myga.GeneticAlgorithm;
import moomap.myga.TopoMapEvaluator;
import moomap.myga.TopoMapIndividual;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.grid.Grid2D;
import spawnn.som.utils.SomUtils;

public class CompareJMetalVSMyGA {

	private static Logger log = Logger.getLogger(CompareJMetalVSMyGA.class);

	public static void main(String[] args) {
		try {

			int xDim = 5, yDim = 5;

			Random r = new Random();
			List<double[]> samples = new ArrayList<double[]>();
			for (int i = 0; i < 250; i++)
				samples.add(new double[] { r.nextDouble(), r.nextDouble() });

			final Dist<double[]> dist = new EuclideanDist();
			final BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(dist);

			double obj = 0;
			double qe = 0;
			double pearson = 0;
			double te = 0;
			for (int j = 0; j < 50; j++) {

				Problem problem = new TopoMapSingle(xDim, yDim, samples, bg, dist);
				Algorithm algorithm = new gGA(problem);

				// Number of solutions depends on max avaluations... why?
				algorithm.setInputParameter("populationSize", 40);
				algorithm.setInputParameter("maxEvaluations", 50000);

				HashMap<String, Object> parameters = new HashMap<String, Object>();
				parameters.put("probability", 0.9); // 0.9
				Operator crossover = new TopoMapNBCrossover(parameters);
				// Operator crossover = new TopoMapUniformCrossover(parameters);

				parameters = new HashMap<String, Object>();
				parameters.put("probability", 1.0 / (xDim * yDim));
				Operator mutation = new RndVectorMutation(parameters);

				Operator selection = new BinaryTournament(null);
				algorithm.addOperator("crossover", crossover);
				algorithm.addOperator("mutation", mutation);
				algorithm.addOperator("selection", selection);

				SolutionSet population = algorithm.execute();

				Solution best = null;
				for (int i = 0; i < population.size(); i++) {
					if (best == null || population.get(i).getObjective(0) < best.getObjective(0))
						best = population.get(i);
				}

				obj += best.getObjective(0);
				Grid2D<double[]> grid = (Grid2D<double[]>) ((TopoMap) best.getDecisionVariables()[0]).grid_;

				qe += SomUtils.getMeanQuantError(grid, bg, dist, samples);
				te += SomUtils.getTopoError(grid, bg, samples);
				pearson += SomUtils.getTopoCorrelation(samples, grid, bg, dist,SomUtils.SPEARMAN_TYPE);
			}

			log.info("jmetal: ");
			log.info("obj: " + (obj / 50));
			log.info("qe: " + (qe / 50));
			log.info("te: " + (te / 50));
			log.info("pearson: " + (pearson / 50));

			obj = 0;
			qe = 0;
			pearson = 0;
			te = 0;
			for (int j = 0; j < 50; j++) {
				GeneticAlgorithm ga = new GeneticAlgorithm(new TopoMapEvaluator(samples, bg, dist));

				List<GAIndividual> init = new ArrayList<GAIndividual>();
				for (int i = 0; i < 40; i++)
					init.add(new TopoMapIndividual(xDim, yDim, 2));

				TopoMapIndividual.mutRate = 1.0 / (xDim * yDim);
				ga.recombProb = 0.9;

				GAIndividual best = ga.search(init);

				Grid2D<double[]> grid = ((TopoMapIndividual) best).grid;

				System.out.println(grid);

				obj += best.getValue();
				qe += SomUtils.getMeanQuantError(grid, bg, dist, samples);
				te += SomUtils.getTopoError(grid, bg, samples);
				pearson += SomUtils.getTopoCorrelation(samples, grid, bg, dist,SomUtils.SPEARMAN_TYPE);
			}

			log.info("myga: ");
			log.info("obj: " + (obj / 50));
			log.info("qe: " + (qe / 50));
			log.info("te: " + (te / 50));
			log.info("pearson: " + (pearson / 50));

		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (JMException e) {
			e.printStackTrace();
		}
	}
}
