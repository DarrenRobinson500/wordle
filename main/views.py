from django.shortcuts import render, redirect
from .models import Word
import datetime
from collections import Counter
import graphviz

input_array = []
all_words = ["cigar","rebut","sissy","humph","awake","blush","focal","evade","naval","serve","heath","dwarf","model","karma","stink","grade","quiet","bench","abate","feign","major","death","fresh","crust","stool","colon","abase","marry","react","batty","pride","floss","helix","croak","staff","paper","unfed","whelp","trawl","outdo","adobe","crazy","sower","repay","digit","crate","cluck","spike","mimic","pound","maxim","linen","unmet","flesh","booby","forth","first","stand","belly","ivory","seedy","print","yearn","drain","bribe","stout","panel","crass","flume","offal","agree","error","swirl","argue","bleed","delta","flick","totem","wooer","front","shrub","parry","biome","lapel","start","greet","goner","golem","lusty","loopy","round","audit","lying","gamma","labor","islet","civic","forge","corny","moult","basic","salad","agate","spicy","spray","essay","fjord","spend","kebab","guild","aback","motor","alone","hatch","hyper","thumb","dowry","ought","belch","dutch","pilot","tweed","comet","jaunt","enema","steed","abyss","growl","fling","dozen","boozy","erode","world","gouge","click","briar","great","altar","pulpy","blurt","coast","duchy","groin","fixer","group","rogue","badly","smart","pithy","gaudy","chill","heron","vodka","finer","surer","radio","rouge","perch","retch","wrote","clock","tilde","store","prove","bring","solve","cheat","grime","exult","usher","epoch","triad","break","rhino","viral","conic","masse","sonic","vital","trace","using","peach","champ","baton","brake","pluck","craze","gripe","weary","picky","acute","ferry","aside","tapir","troll","unify","rebus","boost","truss","siege","tiger","banal","slump","crank","gorge","query","drink","favor","abbey","tangy","panic","solar","shire","proxy","point","robot","prick","wince","crimp","knoll","sugar","whack","mount","perky","could","wrung","light","those","moist","shard","pleat","aloft","skill","elder","frame","humor","pause","ulcer","ultra","robin","cynic","aroma","caulk","shake","dodge","swill","tacit","other","thorn","trove","bloke","vivid","spill","chant","choke","rupee","nasty","mourn","ahead","brine","cloth","hoard","sweet","month","lapse","watch","today","focus","smelt","tease","cater","movie","saute","allow","renew","their","slosh","purge","chest","depot","epoxy","nymph","found","shall","stove","lowly","snout","trope","fewer","shawl","natal","comma","foray","scare","stair","black","squad","royal","chunk","mince","shame","cheek","ample","flair","foyer","cargo","oxide","plant","olive","inert","askew","heist","shown","zesty","trash","larva","forgo","story","hairy","train","homer","badge","midst","canny","shine","gecko","farce","slung","tipsy","metal","yield","delve","being","scour","glass","gamer","scrap","money","hinge","album","vouch","asset","tiara","crept","bayou","atoll","manor","creak","showy","phase","froth","depth","gloom","flood","trait","girth","piety","goose","float","donor","atone","primo","apron","blown","cacao","loser","input","gloat","awful","brink","smite","beady","rusty","retro","droll","gawky","hutch","pinto","egret","lilac","sever","field","fluff","agape","voice","stead","berth","madam","night","bland","liver","wedge","roomy","wacky","flock","angry","trite","aphid","tryst","midge","power","elope","cinch","motto","stomp","upset","bluff","cramp","quart","coyly","youth","rhyme","buggy","alien","smear","unfit","patty","cling","glean","label","hunky","khaki","poker","gruel","twice","twang","shrug","treat","waste","merit","woven","needy","clown","irony","ruder","gauze","chief","onset","prize","fungi","charm","gully","inter","whoop","taunt","leery","class","theme","lofty","tibia","booze","alpha","thyme","doubt","parer","chute","stick","trice","alike","recap","saint","glory","grate","admit","brisk","soggy","usurp","scald","scorn","leave","twine","sting","bough","marsh","sloth","dandy","vigor","howdy","enjoy","valid","ionic","equal","floor","catch","spade","stein","exist","quirk","denim","grove","spiel","mummy","fault","foggy","flout","carry","sneak","libel","waltz","aptly","piney","inept","aloud","photo","dream","stale","unite","snarl","baker","there","glyph","pooch","hippy","spell","folly","louse","gulch","vault","godly","threw","fleet","grave","inane","shock","crave","spite","valve","skimp","claim","rainy","musty","pique","daddy","quasi","arise","aging","valet","opium","avert","stuck","recut","mulch","genre","plume","rifle","count","incur","total","wrest","mocha","deter","study","lover","safer","rivet","funny","smoke","mound","undue","sedan","pagan","swine","guile","gusty","equip","tough","canoe","chaos","covet","human","udder","lunch","blast","stray","manga","melee","lefty","quick","paste","given","octet","risen","groan","leaky","grind","carve","loose","sadly","spilt","apple","slack","honey","final","sheen","eerie","minty","slick","derby","wharf","spelt","coach","erupt","singe","price","spawn","fairy","jiffy","filmy","stack","chose","sleep","ardor","nanny","niece","woozy","handy","grace","ditto","stank","cream","usual","diode","valor","angle","ninja","muddy","chase","reply","prone","spoil","heart","shade","diner","arson","onion","sleet","dowel","couch","palsy","bowel","smile","evoke","creek","lance","eagle","idiot","siren","built","embed","award","dross","annul","goody","frown","patio","laden","humid","elite","lymph","edify","might","reset","visit","gusto","purse","vapor","crock","write","sunny","loath","chaff","slide","queer","venom","stamp","sorry","still","acorn","aping","pushy","tamer","hater","mania","awoke","brawn","swift","exile","birch","lucky","freer","risky","ghost","plier","lunar","winch","snare","nurse","house","borax","nicer","lurch","exalt","about","savvy","toxin","tunic","pried","inlay","chump","lanky","cress","eater","elude","cycle","kitty","boule","moron","tenet","place","lobby","plush","vigil","index","blink","clung","qualm","croup","clink","juicy","stage","decay","nerve","flier","shaft","crook","clean","china","ridge","vowel","gnome","snuck","icing","spiny","rigor","snail","flown","rabid","prose","thank","poppy","budge","fiber","moldy","dowdy","kneel","track","caddy","quell","dumpy","paler","swore","rebar","scuba","splat","flyer","horny","mason","doing","ozone","amply","molar","ovary","beset","queue","cliff","magic","truce","sport","fritz","edict","twirl","verse","llama","eaten","range","whisk","hovel","rehab","macaw","sigma","spout","verve","sushi","dying","fetid","brain","buddy","thump","scion","candy","chord","basin","march","crowd","arbor","gayly","musky","stain","dally","bless","bravo","stung","title","ruler","kiosk","blond","ennui","layer","fluid","tatty","score","cutie","zebra","barge","matey","bluer","aider","shook","river","privy","betel","frisk","bongo","begun","azure","weave","genie","sound","glove","braid","scope","wryly","rover","assay","ocean","bloom","irate","later","woken","silky","wreck","dwelt","slate","smack","solid","amaze","hazel","wrist","jolly","globe","flint","rouse","civil","vista","relax","cover","alive","beech","jetty","bliss","vocal","often","dolly","eight","joker","since","event","ensue","shunt","diver","poser","worst","sweep","alley","creed","anime","leafy","bosom","dunce","stare","pudgy","waive","choir","stood","spoke","outgo","delay","bilge","ideal","clasp","seize","hotly","laugh","sieve","block","meant","grape","noose","hardy","shied","drawl","daisy","putty","strut","burnt","tulip","crick","idyll","vixen","furor","geeky","cough","naive","shoal","stork","bathe","aunty","check","prime","brass","outer","furry","razor","elect","evict","imply","demur","quota","haven","cavil","swear","crump","dough","gavel","wagon","salon","nudge","harem","pitch","sworn","pupil","excel","stony","cabin","unzip","queen","trout","polyp","earth","storm","until","taper","enter","child","adopt","minor","fatty","husky","brave","filet","slime","glint","tread","steal","regal","guest","every","murky","share","spore","hoist","buxom","inner","otter","dimly","level","sumac","donut","stilt","arena","sheet","scrub","fancy","slimy","pearl","silly","porch","dingo","sepia","amble","shady","bread","friar","reign","dairy","quill","cross","brood","tuber","shear","posit","blank","villa","shank","piggy","freak","which","among","fecal","shell","would","algae","large","rabbi","agony","amuse","bushy","copse","swoon","knife","pouch","ascot","plane","crown","urban","snide","relay","abide","viola","rajah","straw","dilly","crash","amass","third","trick","tutor","woody","blurb","grief","disco","where","sassy","beach","sauna","comic","clued","creep","caste","graze","snuff","frock","gonad","drunk","prong","lurid","steel","halve","buyer","vinyl","utile","smell","adage","worry","tasty","local","trade","finch","ashen","modal","gaunt","clove","enact","adorn","roast","speck","sheik","missy","grunt","snoop","party","touch","mafia","emcee","array","south","vapid","jelly","skulk","angst","tubal","lower","crest","sweat","cyber","adore","tardy","swami","notch","groom","roach","hitch","young","align","ready","frond","strap","puree","realm","venue","swarm","offer","seven","dryer","diary","dryly","drank","acrid","heady","theta","junto","pixie","quoth","bonus","shalt","penne","amend","datum","build","piano","shelf","lodge","suing","rearm","coral","ramen","worth","psalm","infer","overt","mayor","ovoid","glide","usage","poise","randy","chuck","prank","fishy","tooth","ether","drove","idler","swath","stint","while","begat","apply","slang","tarot","radar","credo","aware","canon","shift","timer","bylaw","serum","three","steak","iliac","shirk","blunt","puppy","penal","joist","bunny","shape","beget","wheel","adept","stunt","stole","topaz","chore","fluke","afoot","bloat","bully","dense","caper","sneer","boxer","jumbo","lunge","space","avail","short","slurp","loyal","flirt","pizza","conch","tempo","droop","plate","bible","plunk","afoul","savoy","steep","agile","stake","dwell","knave","beard","arose","motif","smash","broil","glare","shove","baggy","mammy","swamp","along","rugby","wager","quack","squat","snaky","debit","mange","skate","ninth","joust","tramp","spurn","medal","micro","rebel","flank","learn","nadir","maple","comfy","remit","gruff","ester","least","mogul","fetch","cause","oaken","aglow","meaty","gaffe","shyly","racer","prowl","thief","stern","poesy","rocky","tweet","waist","spire","grope","havoc","patsy","truly","forty","deity","uncle","swish","giver","preen","bevel","lemur","draft","slope","annoy","lingo","bleak","ditty","curly","cedar","dirge","grown","horde","drool","shuck","crypt","cumin","stock","gravy","locus","wider","breed","quite","chafe","cache","blimp","deign","fiend","logic","cheap","elide","rigid","false","renal","pence","rowdy","shoot","blaze","envoy","posse","brief","never","abort","mouse","mucky","sulky","fiery","media","trunk","yeast","clear","skunk","scalp","bitty","cider","koala","duvet","segue","creme","super","grill","after","owner","ember","reach","nobly","empty","speed","gipsy","recur","smock","dread","merge","burst","kappa","amity","shaky","hover","carol","snort","synod","faint","haunt","flour","chair","detox","shrew","tense","plied","quark","burly","novel","waxen","stoic","jerky","blitz","beefy","lyric","hussy","towel","quilt","below","bingo","wispy","brash","scone","toast","easel","saucy","value","spice","honor","route","sharp","bawdy","radii","skull","phony","issue","lager","swell","urine","gassy","trial","flora","upper","latch","wight","brick","retry","holly","decal","grass","shack","dogma","mover","defer","sober","optic","crier","vying","nomad","flute","hippo","shark","drier","obese","bugle","tawny","chalk","feast","ruddy","pedal","scarf","cruel","bleat","tidal","slush","semen","windy","dusty","sally","igloo","nerdy","jewel","shone","whale","hymen","abuse","fugue","elbow","crumb","pansy","welsh","syrup","terse","suave","gamut","swung","drake","freed","afire","shirt","grout","oddly","tithe","plaid","dummy","broom","blind","torch","enemy","again","tying","pesky","alter","gazer","noble","ethos","bride","extol","decor","hobby","beast","idiom","utter","these","sixth","alarm","erase","elegy","spunk","piper","scaly","scold","hefty","chick","sooty","canal","whiny","slash","quake","joint","swept","prude","heavy","wield","femme","lasso","maize","shale","screw","spree","smoky","whiff","scent","glade","spent","prism","stoke","riper","orbit","cocoa","guilt","humus","shush","table","smirk","wrong","noisy","alert","shiny","elate","resin","whole","hunch","pixel","polar","hotel","sword","cleat","mango","rumba","puffy","filly","billy","leash","clout","dance","ovate","facet","chili","paint","liner","curio","salty","audio","snake","fable","cloak","navel","spurt","pesto","balmy","flash","unwed","early","churn","weedy","stump","lease","witty","wimpy","spoof","saner","blend","salsa","thick","warty","manic","blare","squib","spoon","probe","crepe","knack","force","debut","order","haste","teeth","agent","widen","icily","slice","ingot","clash","juror","blood","abode","throw","unity","pivot","slept","troop","spare","sewer","parse","morph","cacti","tacky","spool","demon","moody","annex","begin","fuzzy","patch","water","lumpy","admin","omega","limit","tabby","macho","aisle","skiff","basis","plank","verge","botch","crawl","lousy","slain","cubic","raise","wrack","guide","foist","cameo","under","actor","revue","fraud","harpy","scoop","climb","refer","olden","clerk","debar","tally","ethic","cairn","tulle","ghoul","hilly","crude","apart","scale","older","plain","sperm","briny","abbot","rerun","quest","crisp","bound","befit","drawn","suite","itchy","cheer","bagel","guess","broad","axiom","chard","caput","leant","harsh","curse","proud","swing","opine","taste","lupus","gumbo","miner","green","chasm","lipid","topic","armor","brush","crane","mural","abled","habit","bossy","maker","dusky","dizzy","lithe","brook","jazzy","fifty","sense","giant","surly","legal","fatal","flunk","began","prune","small","slant","scoff","torus","ninny","covey","viper","taken","moral","vogue","owing","token","entry","booth","voter","chide","elfin","ebony","neigh","minim","melon","kneed","decoy","voila","ankle","arrow","mushy","tribe","cease","eager","birth","graph","odder","terra","weird","tried","clack","color","rough","weigh","uncut","ladle","strip","craft","minus","dicey","titan","lucid","vicar","dress","ditch","gypsy","pasta","taffy","flame","swoop","aloof","sight","broke","teary","chart","sixty","wordy","sheer","leper","nosey","bulge","savor","clamp","funky","foamy","toxic","brand","plumb","dingy","butte","drill","tripe","bicep","tenor","krill","worse","drama","hyena","think","ratio","cobra","basil","scrum","bused","phone","court","camel","proof","heard","angel","petal","pouty","throb","maybe","fetal","sprig","spine","shout","cadet","macro","dodgy","satyr","rarer","binge","trend","nutty","leapt","amiss","split","myrrh","width","sonar","tower","baron","fever","waver","spark","belie","sloop","expel","smote","baler","above","north","wafer","scant","frill","awash","snack","scowl","frail","drift","limbo","fence","motel","ounce","wreak","revel","talon","prior","knelt","cello","flake","debug","anode","crime","salve","scout","imbue","pinky","stave","vague","chock","fight","video","stone","teach","cleft","frost","prawn","booty","twist","apnea","stiff","plaza","ledge","tweak","board","grant","medic","bacon","cable","brawl","slunk","raspy","forum","drone","women","mucus","boast","toddy","coven","tumor","truer","wrath","stall","steam","axial","purer","daily","trail","niche","mealy","juice","nylon","plump","merry","flail","papal","wheat","berry","cower","erect","brute","leggy","snipe","sinew","skier","penny","jumpy","rally","umbra","scary","modem","gross","avian","greed","satin","tonic","parka","sniff","livid","stark","trump","giddy","reuse","taboo","avoid","quote","devil","liken","gloss","gayer","beret","noise","gland","dealt","sling","rumor","opera","thigh","tonga","flare","wound","white","bulky","etude","horse","circa","paddy","inbox","fizzy","grain","exert","surge","gleam","belle","salvo","crush","fruit","sappy","taker","tract","ovine","spiky","frank","reedy","filth","spasm","heave","mambo","right","clank","trust","lumen","borne","spook","sauce","amber","lathe","carat","corer","dirty","slyly","affix","alloy","taint","sheep","kinky","wooly","mauve","flung","yacht","fried","quail","brunt","grimy","curvy","cagey","rinse","deuce","state","grasp","milky","bison","graft","sandy","baste","flask","hedge","girly","swash","boney","coupe","endow","abhor","welch","blade","tight","geese","miser","mirth","cloud","cabal","leech","close","tenth","pecan","droit","grail","clone","guise","ralph","tango","biddy","smith","mower","payee","serif","drape","fifth","spank","glaze","allot","truck","kayak","virus","testy","tepee","fully","zonal","metro","curry","grand","banjo","axion","bezel","occur","chain","nasal","gooey","filer","brace","allay","pubic","raven","plead","gnash","flaky","munch","dully","eking","thing","slink","hurry","theft","shorn","pygmy","ranch","wring","lemon","shore","mamma","froze","newer","style","moose","antic","drown","vegan","chess","guppy","union","lever","lorry","image","cabby","druid","exact","truth","dopey","spear","cried","chime","crony","stunk","timid","batch","gauge","rotor","crack","curve","latte","witch","bunch","repel","anvil","soapy","meter","broth","madly","dried","scene","known","magma","roost","woman","thong","punch","pasty","downy","knead","whirl","rapid","clang","anger","drive","goofy","email","music","stuff","bleep","rider","mecca","folio","setup","verso","quash","fauna","gummy","happy","newly","fussy","relic","guava","ratty","fudge","femur","chirp","forte","alibi","whine","petty","golly","plait","fleck","felon","gourd","brown","thrum","ficus","stash","decry","wiser","junta","visor","daunt","scree","impel","await","press","whose","turbo","stoop","speak","mangy","eying","inlet","crone","pulse","mossy","staid","hence","pinch","teddy","sully","snore","ripen","snowy","attic","going","leach","mouth","hound","clump","tonal","bigot","peril","piece","blame","haute","spied","undid","intro","basal","rodeo","guard","steer","loamy","scamp","scram","manly","hello","vaunt","organ","feral","knock","extra","condo","adapt","willy","polka","rayon","skirt","faith","torso","match","mercy","tepid","sleek","riser","twixt","peace","flush","catty","login","eject","roger","rival","untie","refit","aorta","adult","judge","rower","artsy","rural","shave","bobby","eclat","fella","gaily","harry","hasty","hydro","liege","octal","ombre","payer","sooth","unset","unlit","vomit","fanny","fetus","butch","stalk","flack","widow","augur"]

all_words_sm = ['cigar', 'rebut', 'sissy', 'humph', 'awake', 'blush', 'focal', 'evade', 'naval', 'serve', 'heath', 'dwarf', 'model', 'karma', 'stink', 'grade', 'quiet', 'bench', ]
all_words_2 = ['cigar', 'rebut',]
first_entry = "slice"
# first_entry = "brain"
today = datetime.date.today()

# https://medium.com/@owenyin/here-lies-wordle-2021-2027-full-answer-list-52017ee99e86

def load_dates(request):
    # deleted_words = ["slave","hasty", "forgo"]
    deleted_words = []
    date_counter = datetime.datetime(2021, 6, 19)
    for x in all_words:
        record = Word.objects.filter(word=x).first()
        if record is not None:
            record.date = date_counter
            record.save()
        else:
            pass
        if x not in deleted_words:
            date_counter += datetime.timedelta(days=1)

    return redirect("upcoming")

def clear(request):
    input_array.clear()
    return redirect("home")

def initial_load(request):
    for x in all_words:
        existing_record = Word.objects.filter(word=x)
        if existing_record is None:
            Word(word=x).save()

    return redirect("home")

def determine_colours(word,entry):
    result = ["", "Grey", "Grey", "Grey", "Grey", "Grey", ""]
    for position in range(1,6):
        colour_short = "X"
        for x in range(0, 5):
            if entry[position-1] == word[x]:
                result[position] = 'Orange'
                colour_short = "O"
        if entry[position-1] == word[position-1]:
            result[position] = "Green"
            colour_short = "G"
        result[6] += colour_short
    return result

def initial_word_distribution(request):
    entries = all_words
    # entries = all_words_sm
    distributions = []
    for entry in entries:
        list_of_results = []
        for x in all_words:
            result = determine_colours(x,entry)[6]
            list_of_results.append(result)
        to_append = (entry, Counter(list_of_results)['XXXXX'])
        distributions.append(to_append)

    distributions = sorted(distributions, key=lambda x: x[1])

    context = {"distributions": distributions}
    return render(request, 'distribution.html', context)


def find_available_words(input_array):
    words = []
    for word in all_words:
        valid = True
        for attempt in input_array:
            count = 0
            for letter in attempt:
                if not check(count, letter[1], letter[0], word, attempt): valid = False
                count += 1
        if valid:
            words.append(word)
    return words

def find_word(letter_array):
    max = 0
    max_word = ""
    for word in all_words:
        count = 0
        for letter in letter_array:
            found = False
            for letter2 in word:
                if letter == letter2: found = True
            if found: count += 1
        if count > max:
            max = count
            max_word = word
    return max_word

def count_same_letters(words):
    first_word = words[0]
    same_letters = 0
    for x in range(0, 5):
        same = True
        for word in words:
            if word[x] != first_word[x]: same = False
        if same: same_letters += 1
    return same_letters

def different_letters(words):
    first_word = words[0]
    different_letters = []
    for x in range(0, 5):
        different = False
        for word in words:
            if word[x] != first_word[x]: different = True
        if different: different_letters.append(x)
    return different_letters

def potential_letters(words, pos):
    potential_letters=[]
    for word in words:
        potential_letters.append(word[pos])
    potential_letters = set(potential_letters)
    return potential_letters

def find_fav_word_old(words):
    max_score = 0
    max_word = ""
    for current_word in words:
        score = 0
        for x in words:
            # if x is not None and len(x) >= 5 and len(current_word) >= 5:
            for y in range(0,5):
                if current_word[y] == x[y]: score += 1
        if score > max_score:
            max_score = score
            max_word = current_word

    # Check if there are a large number of similar words
    if 2 < len(words) < 12:
        if count_same_letters(words) >= 3:
            letters = potential_letters(words, different_letters(words)[0])
            max_word = find_word(letters)
    return max_word


def solve_all(request, attempts):
    if attempts == "X":
        selected_words = Word.objects.filter(attempts__isnull=True)
    elif attempts == "all":
        selected_words = Word.objects.all()
    else:
        selected_words = Word.objects.filter(attempts=attempts)

    for x in selected_words:
        test_logic(x.word)

    return redirect("summary")

def test(request, word):
    context = test_logic(word)
    return render(request, 'test.html', context)

def test_logic(word):
    word = word[0:5]
    input_array.clear()
    found = False
    record = Word.objects.filter(word=word).first()

    # Round 1
    entry = first_entry
    colours = determine_colours(word,entry)
    record.outcome1 = colours[6]
    input = [(entry[0], colours[1]), (entry[1], colours[2]), (entry[2], colours[3]), (entry[3], colours[4]), (entry[4], colours[5]), ]
    input_array.append(input)
    words1 = find_available_words(input_array)
    words1.sort()
    count1 = len(words1)
    fav_word1 = find_fav_word(words1)
    if word == entry: found = True

    # Round 2
    entry = fav_word1
    colours = determine_colours(word,entry)
    record.outcome2 = colours[6]
    input = [(entry[0], colours[1]), (entry[1], colours[2]), (entry[2], colours[3]), (entry[3], colours[4]), (entry[4], colours[5]), ]
    input_array.append(input)
    words2 = find_available_words(input_array)
    words2.sort()
    count2 = len(words2)
    fav_word2 = find_fav_word(words2)
    if fav_word1 == word: found = True

    # Round 3
    entry = fav_word2
    colours = determine_colours(word,entry)
    record.outcome3 = colours[6]
    input = [(entry[0], colours[1]), (entry[1], colours[2]), (entry[2], colours[3]), (entry[3], colours[4]), (entry[4], colours[5]), ]
    input_array.append(input)
    words3 = find_available_words(input_array)
    words3.sort()
    count3 = len(words3)
    fav_word3 = find_fav_word(words3)
    if fav_word2 == word: found = True

    # Round 4
    entry = fav_word3
    colours = determine_colours(word,entry)
    record.outcome4 = colours[6]
    input = [(entry[0], colours[1]), (entry[1], colours[2]), (entry[2], colours[3]), (entry[3], colours[4]), (entry[4], colours[5]), ]
    if not found: input_array.append(input)
    words4 = find_available_words(input_array)
    words4.sort()
    count4 = len(words4)
    fav_word4 = find_fav_word(words4)
    if fav_word3 == word: found = True

    # Round 5
    entry = fav_word4
    colours = determine_colours(word,entry)
    record.outcome5 = colours[6]
    input = [(entry[0], colours[1]), (entry[1], colours[2]), (entry[2], colours[3]), (entry[3], colours[4]), (entry[4], colours[5]), ]
    if not found: input_array.append(input)
    words5 = find_available_words(input_array)
    words5.sort()
    count5 = len(words5)
    fav_word5 = find_fav_word(words5)
    if fav_word4 == word: found = True

    # Round 6
    entry = fav_word5
    colours = determine_colours(word,entry)
    record.outcome6 = colours[6]
    input = [(entry[0], colours[1]), (entry[1], colours[2]), (entry[2], colours[3]), (entry[3], colours[4]), (entry[4], colours[5]), ]
    if not found: input_array.append(input)
    words6 = find_available_words(input_array)
    words6.sort()
    count6 = len(words6)
    fav_word6 = find_fav_word(words6)

    if word == first_entry: result = 1
    elif fav_word1 == word: result = 2
    elif fav_word2 == word: result = 3
    elif fav_word3 == word: result = 4
    elif fav_word4 == word: result = 5
    elif fav_word5 == word: result = 6
    else: result = -1

    context = {'word':word.upper(), 'entry': entry, 'input_array': input_array, 'result': result, 'record': record,
               'words1': words1, 'count1':count1, 'fav_word1': fav_word1,
               'words2': words2, 'count2':count2, 'fav_word2': fav_word2,
               'words3': words3, 'count3':count3, 'fav_word3': fav_word3,
               'words4': words4, 'count4':count4, 'fav_word4': fav_word4,
               'words5': words5, 'count5':count5, 'fav_word5': fav_word5,
               'words6': words6, 'count6':count6, 'fav_word6': fav_word6,
               }

    if first_entry == "brain":
        record.attempts_brain = result
    else:
        record.attempts = result
    record.save()

    return context

def summary(request, outcome1=None):
    words = Word.objects.all()
    if outcome1: words = words.filter(outcome1=outcome1)
    attempt1 = words.filter(attempts=1)
    attempt2 = words.filter(attempts=2)
    attempt3 = words.filter(attempts=3)
    attempt4 = words.filter(attempts=4)
    attempt5 = words.filter(attempts=5)
    attempt6 = words.filter(attempts=6)
    attemptNR = words.filter(attempts=-1).order_by('outcome1')
    attemptNone = words.filter(attempts__isnull=True)
    score = (attempt1.count() + 2 * attempt2.count() + 3 * attempt3.count() + 4 * attempt4.count() + 5 * attempt5.count() + 6 * attempt6.count() + 7 * attemptNR.count()) / (attempt1.count() + attempt2.count() + attempt3.count() + attempt4.count() + attempt5.count() + attempt6.count() + attemptNR.count())
    score = round(score,3)
    entry_record = Word.objects.filter(word=first_entry).first()
    if entry_record:
        entry_record.first_entry_score = score
        entry_record.save()

    context = {'score': score, 'attempt1':attempt1, 'attempt2':attempt2, 'attempt3':attempt3, 'attempt4':attempt4,
               'attempt5': attempt5, 'attempt6':attempt6, 'attemptNR': attemptNR, 'attemptNone': attemptNone, "first_entry":first_entry}

    return render(request, 'summary.html', context)

def potential_outcomes(word, words,attempt_array):
    return

def find_fav_word(words):
    if not words: return None
    result = {}
    min_result = len(words)
    fav_word = words[0]
    for entry in words:
        colour_array = []
        for x in words:
            colours = determine_colours(x,entry)[6] # 6 is the shortened version of the color array
            colour_array.append(colours)
        count = Counter(colour_array).most_common(1)[0][1]
        if count < min_result:
            min_result = count
            fav_word = entry
        result[entry] = count
    # Check if there are a large number of similar words
    if 2 < len(words) < 12:
        if count_same_letters(words) >= 3:
            letters = potential_letters(words, different_letters(words)[0])
            fav_word = find_word(letters)

    return fav_word

def sandpit(request):

    # words = Word.objects.filter(word="saint") | Word.objects.filter(word="stink")
    # words = Word.objects.filter(outcome1="GXGXX")

    # words = Word.objects.all()
    words = Word.objects.filter(date__gte=today)
    words = words.filter(outcome1="GXXXG")
    outcomes_list = []
    for word in words:
        outcomes_list.append(word.outcome1)
    outcomes = set(outcomes_list)
    svg_files = []

    for outcome1 in outcomes:
        dot = graphviz.Digraph(comment="Graphic", format='svg', graph_attr={'concentrate': 'true', })
        dot.attr(rankdir='LR')
        dot.strict = True
        group_dict = {}
        attempts = [1,2,3,4,5,6,"X"]
        for x in attempts:
            name = 'cluster_' + str(x)
            group_dict[name] = graphviz.Digraph(name)
            group_dict[name].attr(style="filled", labeljust="l", label=name)
            dot.subgraph(group_dict[name])

        words = Word.objects.filter(outcome1=outcome1)
        print(outcome1,words.count())
        for word in words:
            outcomes = [word.outcome1, word.outcome2, word.outcome3, word.outcome4, word.outcome5, word.outcome6]
            first = True
            count = 1
            finished = False
            for outcome in outcomes:
                if not finished:
                    cluster_name = 'cluster_'+ str(count)
                    if first:
                        group_dict[cluster_name].node(outcome, shape='rect')
                        first = False
                    else:
                        if outcome == "GGGGG":
                            group_dict[cluster_name].node(word.word, shape='rect', style='filled, rounded', fontcolor='red', color='red')
                            dot.edge(previous_outcome + str(count-1), word.word)
                            finished = True
                        else:
                            group_dict[cluster_name].node(outcome + str(count), shape='rect', style='filled, rounded', fontcolor='red', color='red')
                            dot.edge(previous_outcome + str(count-1), outcome + str(count))

                    previous_outcome = outcome
                    count += 1
            if not finished: # this is unsolved case
                print(word.word)
                group_dict["cluster_X"].node(word.word, shape='rect', style='filled, rounded', fontcolor='red', color='red')
                dot.edge(previous_outcome + str(count - 1), word.word)

        dot.format = "svg"
        dot.render('media/files/picture'+outcome1).replace('\\', '/')
        svg_file = "picture" + outcome1
        svg_files.append(svg_file)

    return render(request, "sandpit.html", {'svg_files': svg_files})


def home(request, entry=None):
    # INITIALISE
    if entry: input_array.clear()
    value = ["", "", "", "", "", ]
    colour = ["", "", "", "", "", ]

    # LOAD USER ENTRY
    if request.method == 'POST':
        # for x in request.POST:
        numbers = ["1", "2", "3", "4", "5", ]
        for x in numbers:
            value[int(x)-1] = request.POST.__getitem__(x).lower()
            try: colour[int(x)-1] = request.POST.__getitem__(str(x) + 'x')
            except: colour[x] = ""

        # LOAD VALUES INTO INPUT AND INPUT_ARRAY
        input = []
        for x in range(5):
            input.append((value[x], colour[x]))
        valid = True
        for x, y in input:
            if x == "" or y == "": valid = False
        if valid:
            input_array.append(input)

    words = []  # Potential remaining words
    fav_word = ""

    if len(input_array) > 0: # This condition means you don't see all possible words on your first attempt
        for word in all_words:
            valid = True
            for attempt in input_array:
                count = 0
                for letter, colour in attempt:
                    if not check(count, colour, letter, word, attempt): valid = False
                    count += 1

            if valid:
                record = Word.objects.filter(word=word).first()
                if record.date >= datetime.date.today():
                    words.append(str(record))

        fav_word = find_fav_word(words)
        words.sort()
        count = len(words)
    else:
        count = Word.objects.filter(date__gte=today).count()

    green1 = ""
    green2 = ""
    green3 = ""
    green4 = ""
    green5 = ""
    entry = list("slice")
    if fav_word:
        entry = list(fav_word)

    svg_file = None
    if input_array:
        outcome1 = shorten(input_array[0])
        svg_file = "picture" + outcome1

    numbers = ["1x","2x","3x","4x","5x",]
    context = {'words': words, 'numbers': numbers, 'fav_word': fav_word, 'count': count, 'input_array': input_array,
               'entry':entry, 'green1': green1, 'green2': green2, 'green3': green3, 'green4': green4, 'green5': green5,
               'svg_file': svg_file}
    return render(request, 'home.html', context)

def shorten(outcome):
    result = ""
    for x in outcome:
        result_temp = x[1]
        if result_temp == "Grey": result_temp = "X"
        else: result_temp = result_temp[0]
        result += result_temp
    return result

def check(position, colour, value, word, attempt):
    if colour == "Green" and word[position] != value: return False
    if colour == "Grey":
        found = False
        for x in range(0,5):
            if word[x] == value:
                found = True
                if x != position and attempt[x][1] == "Green": found = False # It was found but green elsewhere
        if found:
            return False
    if colour == "Orange":
        found = False
        for x in range(0,5):
            if word[x] == value: found = True
        if word[position] == value: found = False
        if not found: return False
    return True

def upcoming(request):

    date_counter = datetime.date.today()
    words = []
    for x in range(10):
        record = Word.objects.filter(date=date_counter).first()
        if x == 0:
            words.append(("Today", record.word.upper(), record))
        elif x == 1:
            words.append(("Tomorrow", record.word.upper(), record))
        else:
            date_string = date_counter.strftime('%d %b %Y')
            words.append((date_string, record.word.upper(), record))
        date_counter += datetime.timedelta(days=1)

    context = {'words': words}
    return render(request, 'upcoming.html', context)

