# #!/bin/bash

# # Copyright 2017 Xingyu Na
# # Apache 2.0

# . ./path.sh || exit 1;

# if [ $# != 2 ]; then
#   echo "Usage: $0 <audio-path> <text-path>"
#   echo " $0 /export/a05/xna/data/data_aishell/wav /export/a05/xna/data/data_aishell/transcript"
#   exit 1;
# fi

# # This is the path to the transcript file for the dataset
# aishell_audio_dir=$1
# aishell_text=$2/aishell_transcript_v0.8.txt

# train_dir=data/local/train
# dev_dir=data/local/dev
# test_dir=data/local/test
# tmp_dir=data/local/tmp

# mkdir -p $train_dir
# mkdir -p $dev_dir
# mkdir -p $test_dir
# mkdir -p $tmp_dir

# # data directory check
# if [ ! -d $aishell_audio_dir ] || [ ! -f $aishell_text ]; then
#   echo "Error: $0 requires two directory arguments"
#   exit 1;
# fi

# # find wav audio file for train, dev and test resp.
# find $aishell_audio_dir -iname "*.wav" > $tmp_dir/wav.flist

# # Count and check that the number of files matches the expected number
# # n=`cat $tmp_dir/wav.flist | wc -l`
# # [ $n -ne 141925 ] && \
# #   echo Warning: expected 141925 data data files, found $n

# grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
# grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
# grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

# rm -r $tmp_dir

# # Transcriptions preparation
# for dir in $train_dir $dev_dir $test_dir; do
#   echo Preparing $dir transcriptions
#   sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
#   sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
#   paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
#   utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
#   awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
#   utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
#   utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
#   sort -u $dir/transcripts.txt > $dir/text
#   utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
# done

# mkdir -p data/train data/dev data/test

# for f in spk2utt utt2spk wav.scp text; do
#   cp $train_dir/$f data/train/$f || exit 1;
#   cp $dev_dir/$f data/dev/$f || exit 1;
#   cp $test_dir/$f data/test/$f || exit 1;
# done

# echo "$0: AISHELL data preparation succeeded"
# exit 0;

if (@ARGV != 4) {
  print STDERR "Usage: $0 <path-to-commonvoice-corpus> <path-to-meta> <dataset> <out-dir>\n";
  print STDERR "e.g. $0 /export/data/cv_en_1488h_20191210 data train data/train\n";
  exit(1);
}

($db_base, $meta_base, $dataset, $out_dir) = @ARGV;
mkdir $out_dir unless -d $out_dir;

open(TSV, "<", "$meta_base/$dataset.tsv") or die "cannot open dataset TSV file";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(GNDR,">", "$out_dir/utt2gender") or die "Could not open the output file $out_dir/utt2gender";
open(LANG,">", "$out_dir/utt2lang") or die "Could not open the output file $out_dir/utt2lang";
open(TEXT,">", "$out_dir/text") or die "Could not open the output file $out_dir/text";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
my $header = <TSV>;
while(<TSV>) {
  chomp;
  ($client_id, $filepath, $text, $upvotes, $downvotes, $age, $gender, $accent) = split("\t", $_);
  # TODO: these are empty files, ideally should exclude before sampling data
  next if $filepath =~ /common_voice_en_194119(6[5-8]|7[4-8]|8[4-9]|9[02])\.mp3/;
  # TODO: gender information is probably not used anywhere?
  if ($gender eq "female") {
    $gender = "f";
  } else {
    # Use male as default if not provided (no reason, just adopting the same default as in voxforge)
    $gender = "m";
  }
  # Assume client ID uniquely identifies speakers (i.e. nobody shared a recording device)
  $spkr = $client_id;
  # Prefix filename with client ID so that everything sorts together as it should
  # n.b. these are VERY LONG without creating a new mapping from client ID to snappier speaker ID
  $uttId = "$client_id-$filepath";
  $uttId =~ s/\.mp3//g;
  $uttId =~ tr/\//-/;
  print TEXT "$uttId"," ","$text","\n";
  print GNDR "$uttId"," ","$gender","\n";
  print LANG "$uttId"," ","$accent","\n";
  # This will be read as a Kaldi pipe to downsample audio
  print WAV "$uttId"," sox $db_base/clips/$filepath -t wav -r 16k -b 16 -e signed - |\n";
  print SPKR "$uttId"," $spkr","\n";
}
close(SPKR) || die;
close(TEXT) || die;
close(WAV)  || die;
close(GNDR) || die;
close(LANG) || die;
close(WAVLIST);

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2lang >$out_dir/lang2utt") != 0) {
  die "Error creating lang2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}



