import train, predict
import tensorflow as tf

def main(_) :
    # train
    train.run(False)

    # eval
    train.run(True)

    # predict
    test_txt = " once again the specialists were not able to handle the imbalances on the floor of the new york stock exchange said christopher " \
               "<unk> senior vice president at <unk> securities corp <unk> james <unk> chairman of specialists henderson brothers inc. it is easy " \
               "to say the specialist is n't doing his job when the dollar is in a <unk> even central banks ca n't stop it speculators are calling for " \
               "a degree of liquidity that is not there in the market many money managers and some traders had already left their offices early friday " \
               "afternoon on a warm autumn day because the stock market was so quiet then in a <unk> plunge the dow jones industrials in barely an hour " \
               "surrendered about a third of their gains this year <unk> up a 190.58-point or N N loss on the day in <unk> trading volume <unk> trading" \
               " accelerated to N million shares a record for the big board " \
               "accelerated what?"
    predict.run(test_txt)

if __name__ == '__main__':
    tf.app.run()