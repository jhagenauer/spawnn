package spawnn.gui;

import java.awt.CardLayout;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class DistMatrixDialog extends JDialog implements ActionListener {

	private static final long serialVersionUID = 2783445681242978907L;
	JButton save, cancel;
	
	JPanel cards;
	JComboBox cb;
	JTextField power, knns;
	JCheckBox rowNorm;
	File file;

	public static enum DistMatType {InvDistance, kNN};
	
	private enum Card {
		one, two
	};
	
	public DistMatrixDialog(Frame parent, String string, boolean b ) {
		super(parent, string, b);
		
		setLayout(new MigLayout(""));

		add(new JLabel("Type:"));
		cb = new JComboBox();
		cb.setModel(new DefaultComboBoxModel(DistMatType.values()));
		cb.addActionListener(this);
		add(cb, "wrap");
		
		JPanel jp_1 = new JPanel(new MigLayout());
		jp_1.add(new JLabel("Power:"));
		power = new JTextField("1", 3);
		jp_1.add(power, "wrap");

		JPanel jp_2 = new JPanel(new MigLayout());
		jp_2.add(new JLabel("k:"));
		knns = new JTextField("5", 3);
		jp_2.add(knns, "wrap");
		
		cards = new JPanel(new CardLayout());
		cards.add(jp_1, Card.one.toString());
		cards.add(jp_2, Card.two.toString());
		
		add(cards, "span 2, wrap");
		
		add( new JLabel("Row-normalize:"));
		rowNorm = new JCheckBox();
		add( rowNorm, "wrap" );

		save = new JButton("Save...");
		save.addActionListener(this);
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
		add(save, "span2, split 2, push");
		add(cancel);

		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		pack();
		setAlwaysOnTop(true);
		setLocationRelativeTo(parent);
		setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == cb) {
			CardLayout cl = (CardLayout) (cards.getLayout());
			if (cb.getSelectedItem() == DistMatType.InvDistance )
				cl.show(cards, Card.one.toString());
			else
				cl.show(cards, Card.two.toString());
		} else if (e.getSource() == save) {
			JFileChooser fc = new JFileChooser();
			int state = fc.showSaveDialog(this);
			if( state == JFileChooser.APPROVE_OPTION )
				file = fc.getSelectedFile(); 
			else
				file = null;
			dispose();
		} else {
			dispose();
		}
	}
	
	public DistMatType getDMType() {
		return (DistMatType) cb.getSelectedItem();
	}
	
	public int getKNNs() {
		return Integer.parseInt(knns.getText());
	}
	
	public double getPower() {
		return Double.parseDouble(power.getText());
	}
	
	public boolean rowNorm() {
		return rowNorm.isSelected();
	}
	
	public File getFile() {
		return file;
	}
}
